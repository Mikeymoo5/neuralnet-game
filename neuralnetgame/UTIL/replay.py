import time
import ReplayMemory
from Environments.SimEnvironment import SimEnv
def viewReplay(replay_memory: ReplayMemory, env: SimEnv):
    replay_memory.loadMemory("memory.txt")
    memory = replay_memory.memory
    for i in range(len(memory)):
        state = memory[i].state
        env.setState()
        time.sleep(0.5)
