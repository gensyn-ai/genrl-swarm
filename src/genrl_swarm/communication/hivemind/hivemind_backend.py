import os
import pickle
import time
import inspect
from typing import Any, Dict, List, Optional

import torch.distributed as dist
from hivemind import DHT, get_dht_time

from genrl_swarm.communication.communication import Communication
from genrl_swarm.serialization.game_tree import from_bytes, to_bytes


class HivemindRendezvouz:
    _STORE = None
    _IS_MASTER = False
    _IS_LAMBDA = False

    @classmethod
    def init(cls, is_master: bool = False):
        cls._IS_MASTER = is_master
        cls._IS_LAMBDA = os.environ.get("LAMBDA", False)
        if cls._STORE is None and cls._IS_LAMBDA:
            world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
            cls._STORE = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=int(os.environ["MASTER_PORT"]),
                is_master=is_master,
                world_size=world_size,
                wait_for_workers=True,
            )

    @classmethod
    def is_bootstrap(cls) -> bool:
        return cls._IS_MASTER

    @classmethod
    def set_initial_peers(cls, initial_peers):
        pass
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        if cls._IS_LAMBDA:
            cls._STORE.set("initial_peers", pickle.dumps(initial_peers))

    @classmethod
    def get_initial_peers(cls):
        if cls._STORE is None and cls._IS_LAMBDA:
            cls.init()
        cls._STORE.wait(["initial_peers"])
        peer_bytes = cls._STORE.get("initial_peers")
        initial_peers = pickle.loads(peer_bytes)
        return initial_peers


class HivemindBackend(Communication):
    # Constants for configuration
    DEFAULT_BOOTSTRAP_WAIT = 2.0
    DEFAULT_DHT_READY_TIMEOUT = 10.0
    DEFAULT_READY_CHECK_INTERVAL = 0.5
    DEFAULT_SHUTDOWN_TIMEOUT = 5.0
    
    # Known safe parameters for DHT initialization
    SAFE_DHT_PARAMS = frozenset({
        'cache_locally', 'cache_on_store', 'identity', 'host_maddrs',
        'announce_maddrs', 'use_ipfs', 'record_validators', 'protocol_version'
    })

    def __init__(
        self,
        initial_peers: Optional[List[str]] = None,
        timeout: int = 600,
        disable_caching: bool = False,  
        beam_size: int = 1000,
        bootstrap_wait_time: float = DEFAULT_BOOTSTRAP_WAIT,
        dht_ready_timeout: float = DEFAULT_DHT_READY_TIMEOUT,
        **kwargs,
    ):
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.bootstrap = HivemindRendezvouz.is_bootstrap()
        self.beam_size = beam_size 
        self.bootstrap_wait_time = bootstrap_wait_time
        self.dht_ready_timeout = dht_ready_timeout
        self.dht = None
        
        if disable_caching:
            kwargs.update({'cache_locally': False, 'cache_on_store': False})
        
        self._initialize_dht(initial_peers, kwargs)
        self.step_ = 0

    def _get_dht_configs(self) -> List[Dict[str, Any]]:
        """Get ordered list of DHT configurations from safest to most permissive"""
        return [
            # Config 1: Local loopback (most compatible)
            {"host_maddrs": ["/ip4/127.0.0.1/tcp/0"]},
            # Config 2: System auto-selection
            {},
            # Config 3: Local loopback without await_ready
            {"host_maddrs": ["/ip4/127.0.0.1/tcp/0"], "await_ready": False},
            # Config 4: All interfaces TCP only
            {"host_maddrs": ["/ip4/0.0.0.0/tcp/0"]},
            # Config 5: Full configuration as fallback
            {
                "host_maddrs": ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                "await_ready": False
            }
        ]

    def _filter_safe_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to only include known safe parameters"""
        return {k: v for k, v in kwargs.items() if k in self.SAFE_DHT_PARAMS}

    def _initialize_dht(self, initial_peers: Optional[List[str]], kwargs: Dict[str, Any]):
        """Initialize DHT with fallback configurations"""
        safe_kwargs = self._filter_safe_kwargs(kwargs)
        dht_configs = self._get_dht_configs()
        last_error = None
        
        for i, base_config in enumerate(dht_configs):
            try:
                final_config = {**safe_kwargs, **base_config}
                
                if self.bootstrap:
                    self._create_bootstrap_dht(initial_peers, final_config)
                else:
                    self._create_worker_dht(initial_peers, final_config)
                
                return  # Success, exit early
                
            except Exception as e:
                last_error = e
                self._cleanup_dht()
                
                if i < len(dht_configs) - 1:
                    time.sleep(1)  # Brief pause between retries
                    continue
        
        raise RuntimeError(f"All DHT configurations failed. Last error: {last_error}")

    def _create_bootstrap_dht(self, initial_peers: Optional[List[str]], config: Dict[str, Any]):
        """Create and initialize bootstrap DHT node"""
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers or [],
            **config,
        )
        
        if self._wait_for_dht_ready():
            try:
                dht_maddrs = self.dht.get_visible_maddrs(latest=True)
                HivemindRendezvouz.set_initial_peers(dht_maddrs)
            except Exception:
                pass  # Continue even if address retrieval fails

    def _create_worker_dht(self, initial_peers: Optional[List[str]], config: Dict[str, Any]):
        """Create and initialize worker DHT node"""
        initial_peers = initial_peers or HivemindRendezvouz.get_initial_peers()
        
        self.dht = DHT(
            start=True,
            initial_peers=initial_peers,
            **config,
        )
        
        self._wait_for_dht_ready()

    def _wait_for_dht_ready(self) -> bool:
        """Wait for DHT to be ready using intelligent polling"""
        max_attempts = int(self.dht_ready_timeout / self.DEFAULT_READY_CHECK_INTERVAL)
        
        for attempt in range(max_attempts):
            try:
                self.dht.get_visible_maddrs(latest=True)
                return True
            except Exception:
                if attempt < max_attempts - 1:
                    time.sleep(self.DEFAULT_READY_CHECK_INTERVAL)
        
        # Fallback to simple wait if polling fails
        fallback_wait = self.bootstrap_wait_time if self.bootstrap else 1.0
        time.sleep(fallback_wait)
        return False

    def _cleanup_dht(self):
        """Comprehensive DHT cleanup to prevent resource leaks"""
        if not self.dht:
            return
            
        try:
            self._graceful_shutdown()
        except Exception:
            self._force_cleanup()
        finally:
            self.dht = None

    def _graceful_shutdown(self):
        """Attempt graceful DHT shutdown with timeout if supported"""
        if not hasattr(self.dht, 'shutdown'):
            return
            
        sig = inspect.signature(self.dht.shutdown)
        if 'timeout' in sig.parameters:
            self.dht.shutdown(timeout=self.DEFAULT_SHUTDOWN_TIMEOUT)
        else:
            self.dht.shutdown()

    def _force_cleanup(self):
        """Force cleanup of DHT components when graceful shutdown fails"""
        cleanup_actions = [
            ('_server', 'stop'),
            ('_background_thread', 'join'),
            ('_p2p', 'stop'),
            ('_networking', 'shutdown'),
        ]
        
        for attr_name, method_name in cleanup_actions:
            try:
                component = getattr(self.dht, attr_name, None)
                if component and hasattr(component, method_name):
                    method = getattr(component, method_name)
                    if method_name == 'join':
                        method(timeout=1.0)  # Special case for thread join
                    else:
                        method()
            except Exception:
                continue  # Best effort cleanup

    def all_gather_object(self, obj: Any) -> Dict[str, Any]:
        """Collect objects from all nodes in the swarm"""
        key = str(self.step_)
        try:
            self.dht.get_visible_maddrs(latest=True)  # Connectivity check
            obj_bytes = to_bytes(obj)
            
            self.dht.store(
                key,
                subkey=str(self.dht.peer_id),
                value=obj_bytes,
                expiration_time=get_dht_time() + self.timeout,
                beam_size=self.beam_size,
            )
            
            # Wait briefly for propagation
            time.sleep(1)
            
            # Poll for all responses with timeout
            start_time = time.monotonic()
            while time.monotonic() - start_time < self.timeout:
                output, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                if len(output) >= self.world_size:
                    break
                time.sleep(0.1)  # Small delay between polls
            else:
                raise RuntimeError(
                    f"Failed to obtain {self.world_size} values for {key} within timeout"
                )
            
            self.step_ += 1
            
            # Sort and return results
            sorted_results = sorted(
                [(subkey, from_bytes(value.value)) for subkey, value in output.items()],
                key=lambda x: x[0],
            )
            return dict(sorted_results)
            
        except (BlockingIOError, EOFError):
            # Fallback for network issues
            return {str(self.dht.peer_id): obj}

    def get_id(self) -> str:
        return str(self.dht.peer_id)

    def __del__(self):
        self._cleanup_dht()