import time
from ReplayMemory import Replay
from Environments.SimEnvironment import SimEnv
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pickle
from torch import tensor
from torch import float32
from torch import device
import click
def viewReplay(replay_name: str):
    t_env = gym.make("SimEnv-v0", disable_env_checker=True, render_mode="human") # Disable the environment checker as it throws a warning about flattening the observation, which is done on the next line
    env = FlattenObservation(t_env) # Flatten the observation space
    # replay_memory = Replay(10000)
    replay_path = "replays/"
    replay_fileext = ".rpl"
    f = open(f"{replay_path}{replay_name}{replay_fileext}", 'rb')
    state_mem = pickle.load(f)
    for i in range(len(state_mem)):
        state_tensor = state_mem
        state = state_tensor
        env.setState(state)
        time.sleep(0.5)
