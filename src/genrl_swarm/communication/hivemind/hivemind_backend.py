import os
import pickle
import time
from typing import Any, Dict, List

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
    def __init__(
        self,
        initial_peers: List[str] | None = None,
        timeout: int = 600,
        disable_caching: bool = False,  
        beam_size: int = 1000, 
        **kwargs,
    ):
        self.world_size = int(os.environ.get("HIVEMIND_WORLD_SIZE", 1))
        self.timeout = timeout
        self.bootstrap = HivemindRendezvouz.is_bootstrap()
        self.beam_size = beam_size 
        self.dht = None
        
        if disable_caching:
            kwargs['cache_locally'] = False
            kwargs['cache_on_store'] = False
        
        # Multiple configuration strategies, starting from the safest
        dht_configs = [
            # Config 1: Local loopback address (safest)
            {
                "host_maddrs": ["/ip4/127.0.0.1/tcp/0"]
            },
            # Config 2: Let system auto-select address
            {},
            # Config 3: Local loopback, disable await_ready
            {
                "host_maddrs": ["/ip4/127.0.0.1/tcp/0"],
                "await_ready": False
            },
            # Config 4: Listen on all network interfaces
            {
                "host_maddrs": ["/ip4/0.0.0.0/tcp/0"]
            },
            # Config 5: Original configuration (TCP + UDP)
            {
                "host_maddrs": ["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
                "await_ready": False
            }
        ]
        
        last_error = None
        
        for i, base_config in enumerate(dht_configs):
            try:
                # Only keep known safe parameters, filter out potentially incompatible ones
                safe_kwargs = {}
                known_safe_params = {
                    'cache_locally', 'cache_on_store', 'identity', 'host_maddrs',
                    'announce_maddrs', 'use_ipfs', 'record_validators', 'protocol_version'
                }
                
                for k, v in kwargs.items():
                    if k in known_safe_params:
                        safe_kwargs[k] = v
                
                # Merge configurations
                final_config = {**safe_kwargs, **base_config}
                
                if self.bootstrap:
                    self.dht = DHT(
                        start=True,
                        initial_peers=initial_peers or [],
                        **final_config,
                    )
                    
                    time.sleep(2)  # Wait for bootstrap node to be ready
                    
                    try:
                        dht_maddrs = self.dht.get_visible_maddrs(latest=True)
                        HivemindRendezvouz.set_initial_peers(dht_maddrs)
                    except Exception:
                        pass  # Continue even if getting addresses fails
                    
                else:
                    initial_peers = initial_peers or HivemindRendezvouz.get_initial_peers()
                    
                    self.dht = DHT(
                        start=True,
                        initial_peers=initial_peers,
                        **final_config,
                    )
                    
                    time.sleep(1)  # Wait for connection to establish
                
                break  # Exit on success
                
            except Exception as e:
                last_error = e
                
                # Clean up failed DHT instance
                if self.dht:
                    try:
                        self.dht.shutdown()
                    except:
                        pass
                    finally:
                        self.dht = None
                
                # If not the last config, continue trying
                if i < len(dht_configs) - 1:
                    time.sleep(1)
                    continue
        
        # If all configurations failed
        if self.dht is None:
            raise RuntimeError(f"All DHT configurations failed. Last error: {last_error}")
        
        self.step_ = 0

    def all_gather_object(self, obj: Any) -> Dict[str | int, Any]:
        """Collect objects from all nodes"""
        key = str(self.step_)
        try:
            _ = self.dht.get_visible_maddrs(latest=True)
            obj_bytes = to_bytes(obj)
            self.dht.store(
                key,
                subkey=str(self.dht.peer_id),
                value=obj_bytes,
                expiration_time=get_dht_time() + self.timeout,
                beam_size=self.beam_size,  
            )
            
            time.sleep(1)
            t_ = time.monotonic()
            while True:
                output_, _ = self.dht.get(key, beam_size=self.beam_size, latest=True)
                if len(output_) >= self.world_size:
                    break
                else:
                    if time.monotonic() - t_ > self.timeout:
                        raise RuntimeError(
                            f"Failed to obtain {self.world_size} values for {key} within timeout."
                        )
            self.step_ += 1

            tmp = sorted(
                [(key, from_bytes(value.value)) for key, value in output_.items()],
                key=lambda x: x[0],
            )
            return {key: value for key, value in tmp}
        except (BlockingIOError, EOFError) as e:
            return {str(self.dht.peer_id): obj}

    def get_id(self) -> str:
        """Get node ID"""
        return str(self.dht.peer_id)