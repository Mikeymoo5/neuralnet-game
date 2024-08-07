from collections import namedtuple, deque
import random
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) # Maps state and action to the next state and reward

# Most of this code was provided by the PyTorch tutorial on Reinforcement Learning
class Replay():
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args)) # 'Pushes' the transition to the memory

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) # Chooses a batch_size number of random sample of transitions from the memory
    
    def __len__(self):
        return len(self.memory) # Returns the length of the memory
    
    def getState(self, index):
        return self.memory[index].state

    def loadMemory(self, path_to_file):
        f = open(path_to_file, 'r')
        self.memory = deque([],maxlen=int(f.readline()))