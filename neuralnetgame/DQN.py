import torch.nn as nn
import torch.nn.functional as F

# Much of this code is taken from the Pytorch Reinforcement Learning tutorial - I will modify it soon to be more specific to the game
class DQN(nn.Module):
    def __init__(self, n_actions, n_observations):
        super(DQN, self).__init__() # Calls the nn.Module constructor
        # TODO: Make an easy way to define the number of layers
        self.layer1 = nn.Linear(n_observations, 128) # First layer of the neural net with n_observations inputs and 128 outputs
        self.layer2 = nn.Linear(128, 128) # Creates the second layer of the neural net with 128 inputs and 128 outputs
        self.layer3 = nn.Linear(128, n_actions) # Creates the final layer of the neural net with 128 inputs and n_actions outputs
    def forward(self, x):
        x = F.relu(self.layer1(x)) # Apply the RELU activation function to the first & second layers
        x = F.relu(self.layer2(x)) 
        return self.layer3(x)