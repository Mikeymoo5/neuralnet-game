import time
from ReplayMemory import Replay
from Environments.SimEnvironment import SimEnv
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

def viewReplay(replay_name: str):
    t_env = gym.make("SimEnv-v0", disable_env_checker=True, render_mode="human") # Disable the environment checker as it throws a warning about flattening the observation, which is done on the next line
    env = FlattenObservation(t_env) # Flatten the observation space
    replay_memory = Replay(10000)
    replay_path = "replays/"
    replay_fileext = ".rpl"
    replay_memory.loadMemory(f"{replay_path}{replay_name}{replay_fileext}")
    state_mem = replay_memory.memory
    for i in range(len(state_mem)):
        state = state_mem[i].state
        env.setState()
        time.sleep(0.5)
