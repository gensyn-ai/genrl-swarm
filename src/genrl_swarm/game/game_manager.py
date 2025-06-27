import abc
from enum import Enum
from typing import Any, List, Tuple, Dict, Callable
import time

from genrl_swarm.logging_utils.global_defs import get_logger
from genrl_swarm.state import GameState, GameNode
from genrl_swarm.rewards import RewardManager
from genrl_swarm.trainer import TrainerModule
from genrl_swarm.data import DataManager
from genrl_swarm.communication.communication import Communication
from genrl_swarm.roles import RoleManager #TODO: Implement RoleManager+Pass to game manager
from genrl_swarm.communication import Communication
from genrl_swarm.blockchain import SwarmCoordinator
from genrl_swarm.misc_utils.name_utils import get_name_from_peer_id

# Imports needed only for SwarmGameManager
from collections import defaultdict
import logging 
import os
import sys
from huggingface_hub import login, whoami
from genrl_swarm.communication.hivemind.hivemind_backend import HivemindBackend
from genrl_swarm.logging_utils.system_utils import get_system_info


class RunType(Enum):
    Train = "train"
    Evaluate = "evaluate"
    TrainAndEvaluate = "train_and_evaluate"


class GameManager(abc.ABC): #TODO: Make this use enum
    def __init__(self, 
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication | None = None,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "train",
                 rank: int = 0,
                 **kwargs,
                 ):
        """Initialization method that stores the various managers needed to orchestrate this game"""
        self.state = game_state
        self.rewards = reward_manager
        self.trainer = trainer
        self.data_manager = data_manager
        self.communication = communication or Communication.create(**kwargs)
        self.roles = role_manager
        try:
            self.mode = RunType(run_mode)
        except ValueError:
            get_logger().info(f"Invalid run mode: {run_mode}. Defaulting to train only.")
            self.mode = RunType.Train
        self._rank = rank or self.communication.get_id()
        self.agent_ids = [self._rank] #NOTE: Add more if wanted for game/usecase

    @property
    def rank(self) -> int:
        return self._rank
    
    @rank.setter
    def rank(self, rank: int) -> None:
        self._rank = rank

    @abc.abstractmethod
    def end_of_game(self) -> bool:
        """
        Defines conditions for the game to end and no more rounds/stage should begin. 
        Return True if conditions imply game should end, else False
        """
        pass

    @abc.abstractmethod
    def end_of_round(self) -> bool:
        """
        Defines conditions for end of a round AND no more stages/"turns" should being for this round AND the game state should be reset for stage 0 of your game. 
        Return True if conditions imply game should end and no new round/stage should begin, else False
        """
        pass

    def _hook_after_rewards_updated(self):
        """Hook method called after rewards are updated."""
        pass

    def _hook_after_round_advanced(self):
        """Hook method called after the round is advanced and rewards are reset."""
        pass

    def _hook_after_game(self):
        """Hook method called after the game is finished."""
        pass
    
    #Helper methods
    def aggregate_game_state_methods(self) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
        world_state_pruners = {"environment_pruner": getattr(self, "environment_state_pruner", None),
                               "opponent_pruner": getattr(self, "opponent_state_pruner", None), 
                               "personal_pruner": getattr(self, "personal_state_pruner", None)
                               }
        game_tree_brancher = {"terminal_node_decision_function": getattr(self, "terminal_game_tree_node_decider", None), 
                              "stage_inheritance_function": getattr(self, "stage_inheritance_function", None)
                              }
        return world_state_pruners, game_tree_brancher

    #Core (default) game orchestration methods
    def run_game_stage(self):
        inputs = self.state.get_latest_state() # Fetches the current world state for all agents
        inputs, index_mapping = self.data_manager.prepare_input(inputs, self.state.stage) # Maps game tree states to model ingestable inputs
        outputs = self.trainer.generate(inputs) # Generates a rollout. Ingests inputs indexable in the following way [Agent][Batch Item][Nodes idx within current stage][World state] then outputs something indexable as [Agent][Batch Item][Nodes idx within current stage]
        actions = self.data_manager.prepare_actions(outputs, index_mapping) # Maps model outputs to RL game tree actions
        self.state.append_actions(actions) # Adds the freshly generated rollout to the game state associated with this agent's nodes at this stage

    def run_game_round(self):
        # Loop through stages until end of round is hit
        while not self.end_of_round():
            self.run_game_stage() # Generates rollout and updates the game state
            swarm_payloads = self.communication.all_gather_object(self.state.get_latest_communication()[self.rank])
            world_states = self.data_manager.prepare_states(self.state, swarm_payloads) #Maps states received via communication with the swarm to RL game tree world states
            self.state.advance_stage(world_states) # Prepare for next stage
    
        self.rewards.update_rewards(self.state) # Compute reward functions now that we have all the data needed for this round
        self._hook_after_rewards_updated() # Call hook

        if self.mode in [RunType.Train, RunType.TrainAndEvaluate]:
            self.trainer.train(self.state, self.data_manager, self.rewards) 
        if self.mode in [RunType.Evaluate, RunType.TrainAndEvaluate]:
            self.trainer.evaluate(self.state, self.data_manager, self.rewards)
    
        self.state.advance_round(self.data_manager.get_round_data(), agent_keys=self.agent_ids) # Resets the game state appropriately, stages the next round, and increments round/stage counters appropriatelly
        self.rewards.reset()
        self._hook_after_round_advanced() # Call hook

    def run_game(self):
        # Initialize game and/or run specific details of game state
        world_state_pruners, game_tree_brancher = self.aggregate_game_state_methods()
        self.state._init_game(self.data_manager.get_round_data(), agent_keys=self.agent_ids, world_state_pruners=world_state_pruners, game_tree_brancher=game_tree_brancher) # Prepare game trees within the game state for the initial round's batch of data
        # Loop through rounds until end of the game is hit
        try:
            while not self.end_of_game():
                get_logger().info(f"Starting round: {self.state.round}/{getattr(self, 'max_round', None)}.")
                self.run_game_round() # Loops through stages until end of round signal is received
        except:
            get_logger().exception("Exception occurred during game run.", stack_info=True)
        finally:
            self._hook_after_game()
            self.trainer.cleanup()

class DefaultGameManagerMixin:
    """
    Defines some default behaviour for games with a "shared memory", "linked list" game tree structure, and fixed duration, i.e. the next stage only ever has a single child and all state information from last stage can be "safely" inherited and nodes stop having children at a specific stage.
    """
    #Optional methods
    def environment_state_pruner(self, input: Any) -> Any:
        """
        Optional pruning function for environment states. The format and data types of environment states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the environment state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        return input
    
    def opponent_state_pruner(self, input: Any) -> Any:
        """
        Optional pruning function for opponent states. The format and data types of opponent states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the opponent state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        return input
    
    def personal_state_pruner(self, input: Any) -> Any:
        """
        Optional pruning function for personal states. The format and data types of personal states is game-specific, so exact behaviours should reflect this.
        WARNING: Output of this function is directly set as the personal state of nodes in game tree, which may in turn used for constructing input to your models/agents!
        """
        return input
    
    def terminal_game_tree_node_decider(self, stage_nodes: List[GameNode]) -> List[GameNode]:
        """
        Optional function defining whether the set of nodes from a stage are terminal and, hence, should not be branched.
        Input:
            List[GameNode]: List nodes from a stage in the game.
        Return:
            List[GameNode]: List of nodes from the game tree that should be designated as terminal/leaves. Empty list indicates no nodes should be set as terminal for this tree after said stage.
        """
        terminal = []
        for node in stage_nodes:
            if node["stage"] < self.max_stage-1: #NOTE: For custom terminal functions, you may want to add in your own more complex logic in here for deciding whether a node is terminal. For example, in something like chess, you may want to add logic for checking for checkmates, etc.
                pass
            else:
                terminal.append(node)
        return terminal

    def stage_inheritance_function(self, stage_nodes: List[GameNode]) -> List[List[GameNode]]:
        """
        Optional function defining whether the set of nodes from a stage are terminal and, hence, should not be branched.
        Input:
            List[GameNode]: List nodes from a stage in the game.
        Return:
            List[List[GameNode]]: List of lists of game nodes, where outer-most list contains a list for each node in the input (i.e., stage_nodes) and each inner-list contains children nodes.  
        """
        stage_children = []
        for i, node in enumerate(stage_nodes):
            children = []
            if not node._is_leaf_node(): #NOTE: For custom inheritance functions, you may want to add your own loop in here to generate several children according to whatever logic you desire
                child = GameNode(stage=node.stage+1,
                                 node_idx=0, #Will be overwritten by the game tree if not correct
                                 world_state = node.world_state,
                                 actions=None
                                 )
                children.append(child)
            stage_children.append(children)
        return stage_children
  

class BaseGameManager(DefaultGameManagerMixin, GameManager):
    """
    Basic GameManager with basic functionality baked-in.
    Will end the game when max_rounds is reached, end a round when max_stage is reached.
    """
    def __init__(self,
                 max_stage: int,
                 max_round: int,
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication | None = None,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "Train"
                 ):
        """Init a GameManager which ends the game when max_rounds is reached, ends stage when max_stage is reached, and prunes according to top-k rewards"""
        self.max_stage = max_stage
        self.max_round = max_round
        kwargs = {"game_state": game_state, 
                  "reward_manager": reward_manager, 
                  "trainer": trainer, 
                  "data_manager": data_manager, 
                  "communication": communication,
                  "role_manager": role_manager, 
                  "run_mode": run_mode
                  }
        super().__init__(**kwargs)

    def end_of_game(self) -> bool:
        if self.state.round < self.max_round:
            return False
        else:
            return True
    
    def end_of_round(self) -> bool:
        if self.state.stage < self.max_stage:
            return False
        else:
            return True


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """GameManager that orchestrates a game using a SwarmCoordinator."""
    def __init__(self, 
                 coordinator: SwarmCoordinator, 
                 max_stage: int,
                 max_round: int,
                 game_state: GameState, 
                 reward_manager: RewardManager, 
                 trainer: TrainerModule, 
                 data_manager: DataManager, 
                 communication: Communication,
                 role_manager: RoleManager | None = None,
                 run_mode: str = "train",
                 log_dir: str = "logs",
                 hf_token: str | None = None,
                 hf_push_frequency: int = 20,
                 submit_frequency: int = 3,
                 **kwargs
                 ):

        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode
        )

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31 # 1 month

        #Logging Setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        format_msg = f"[{self.animal_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)

        #Register peer_id and get current round from the chain
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = self.state.round #initialize communication module to contract's round
        self.submit_frequency = submit_frequency

        #enable push to HF if token was provided
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            username = whoami(token=self.hf_token)["name"]
            model_name = self.trainer.model.config.name_or_path.split('/')[-1] 
            model_name += '-Gensyn-Swarm'
            model_name += f"-{self.animal_name}"
            self.trainer.args.hub_model_id = f"{username}/{model_name}"
            self.trainer.args.push_to_hub = True
            self.trainer.args.hub_token = self.hf_token
            self.hf_push_frequency = hf_push_frequency
            get_logger().info('Logging into Hugging Face Hub...')

            login(self.hf_token)

        get_logger().info(f"🐱 Hello 🐈 [{get_name_from_peer_id(self.peer_id)}] 🦮 [{self.peer_id}]!")
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")
        get_logger().info(f"Using Model: {self.trainer.model.config.name_or_path}")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot
        
        return rewards_by_agent

    def _hook_after_rewards_updated(self):
        #submit rewards and winners to the chain
        if self.state.round % self.submit_frequency == 0:
            rewards_by_agent = self._get_total_rewards_by_agent()
            my_rewards = rewards_by_agent[self.peer_id]
            my_rewards = (my_rewards + 1) * (my_rewards > 0) + my_rewards * (my_rewards <= 0)
            self.coordinator.submit_reward(self.state.round, 0, int(my_rewards), self.peer_id)
 
            max_agent, max_rewards = max(rewards_by_agent.items(), key=lambda x: x[1])
            self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)

    def _hook_after_round_advanced(self):
        self._save_to_hf()

        # Block until swarm round advances
        self.agent_block()

    def _hook_after_game(self):
        self._save_to_hf()

    def _save_to_hf(self):
        if self.hf_token not in [None, "None"] and self.state.round % self.hf_push_frequency == 0:
            get_logger().info(f"pushing model to huggingface")
            try:
                repo_id = self.trainer.args.hub_model_id
                if repo_id is None:
                    repo_id = Path(self.trainer.args.output_dir).name
                
                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ]
                )
            except Exception:
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading"
                , stack_info=True)

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = (
            check_interval  # Exponential backoff for already finished rounds.
        )
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            # Retrieve current round and stage.
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(
                    f"🐝 Joining round: {round_num}"
                )
                check_backoff = check_interval  # Reset backoff after successful round
                self.state.round = round_num # advance to swarm's round.
                return
            else:
                get_logger().info(
                    f"Already finished round: {round_num}. Next check in {check_backoff}s."
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                return

        get_logger().info("Training timed out!")
