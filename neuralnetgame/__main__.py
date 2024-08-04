# Project imports
from UTIL import *
import Sprites.Sprites as Sprites
from Environments.SimEnvironment import SimEnv
from DQN import DQN
from ReplayMemory import Replay
from ReplayMemory import Transition

# Misc imports
import pygame
import numpy
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import click
import sys

# Gymnasium and Environment imports
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# Pytorch imports
import torch_directml
from itertools import count
import torch
import torch.nn as nn # Neural network
import torch.optim as optim # Optimizer
import torch.nn.functional as F # Functional

# Pytorch hyperparameters
BATCH_SIZE = 128 # The sample size from the replay memory
GAMMA = 0.99 # Discount factor
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4 # The learning rate of the optimizer

steps_done = 0

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Set the device to use for the neural net
# device = torch_directml.device(
#     "cuda" if torch.cuda.is_available() else 
#     torch_directml.default_device() if torch_directml.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )

device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

click.echo(click.style(f"Using device: {device}", fg="cyan"))

t_env = None
env = None

state, info = None, None
n_observations = None
n_actions = None

policy_net = None
target_net = None

optimizer = None
memory = None

# This function will occasionally choose a random action instead of a predicted action, with the odds decreasing as time goes on
# Taken from the Pytorch Reinforcement Learning tutorial
def select_action(state):
    global env
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# Optimize the model - Taken from the Pytorch Reinforcement Learning tutorial
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

episode_durations = [] # The duration of each episode

# Training loop - Taken from the Pytorch Reinforcement Learning tutorial but modified by me
def train_agent(n_episodes, wait_time=0, render=True, report_interval=10, save_interval=100):
    global env, n_observations, n_actions, policy_net, target_net, optimizer, memory

    t_env = gym.make("SimEnv-v0", disable_env_checker=False) # Disable the environment checker as it throws a warning about flattening the observation, which is done on the next line
    env = FlattenObservation(t_env) # Flatten the observation space

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    policy_net = DQN(n_actions, n_observations).to(device)
    target_net = DQN(n_actions, n_observations).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = Replay(10000) # Create a replay memory with a capacity of 10000

    for i_episode in range(n_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for i in count():
            action = select_action(state) # Selects an action based on the state of the environment
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device) # Converts the reward to a tensor with a single value
            done = terminated or truncated # The episode is done if the environment is terminated or truncated
            if i % report_interval == 0:
                click.echo(click.style(f"Episode {i_episode} - Step {i} - Reward: {reward.item()}", fg="white"))
            if i_episode % save_interval == 0 & i == 0:
                torch.save(policy_net.state_dict(), f"models/model_{i_episode}.pt")
                click.echo(click.style(f"Saving model - Episode {i_episode}", fg="green"))
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Push the transition to the replay memory and progress to the next state
            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model() # Optimize the model

            # Soft update the network weights - I have no clue what this means
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(i + 1)
                break
            if wait_time > 0: time.sleep(wait_time) # Wait for a certain amount of time before taking the next step

def play(model_name="model"):
    try:
        # NOTE: Weights_only is set to true so that only the weights are loaded (duh), preventing malicious code execution from unknown model sources
        agent = torch.load(f"models/{model_name}.pt", weights_only=True)
    except FileNotFoundError:
        click.echo(click.style("Model not found. Exiting...", fg="red"))
        sys.exit(1)
    click.echo(click.style("Model loaded", fg="green"))
    try:
        t_env = gym.make("SimEnv-v0", disable_env_checker=True, render_mode="human") # Disable the environment checker as it throws a warning about flattening the observation, which is done on the next line
        env = FlattenObservation(t_env) # Flatten the observation space
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        sys.exit(1)
    click.echo(click.style("Environment loaded", fg="green"))
    while env.pyg_running:
        time.sleep(1) # Wait for a second before taking the next step
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = policy_net(state).max(1).indices.view(1, 1)
        env.step(action)
        


@click.command()
@click.option("--train", is_flag=True)
@click.option("--train-duration", type=int, default=1000)
@click.option("--report-interval", type=int, default=10)
@click.option("--save-interval", type=int, default=100)
@click.option("--model-name", type=str, default="model")
def __main__(train, train_duration, report_interval, save_interval, model_name) -> None:
    if train:
        click.echo(click.style("Starting in training mode", fg="green"))
        train_agent(train_duration, report_interval=report_interval, save_interval=save_interval)
        # plotting.plot_durations(episode_durations=episode_durations, is_ipython=is_ipython, show_result=True) #TODO: show plot when arg is passed
    else:
        click.echo(click.style("Starting in play mode", fg="green"))
        play(model_name)
        
if __name__ == "__main__":
    __main__()

# OLD CODE - TO BE REMOVED

# # train(1000, report_interval=20)
# # plotting.plot_durations(episode_durations=episode_durations, is_ipython=is_ipython, show_result=True)
# # while True:
# #     pass

# # Misc constants
# GRID_SIZE = 16

# GRID_X = 40
# GRID_Y = 40

# SCREEN_WIDTH = GRID_SIZE * GRID_X
# SCREEN_HEIGHT = GRID_SIZE * GRID_Y
# world = numpy.zeros(shape = (GRID_X, GRID_Y), dtype = int)
# SPRITE_DATA = {
#     "colors": {
#         "pet": (115, 113, 81),
#         "food": (92, 62, 43),
#         "water": (15, 159, 255),
#         "background": (23, 194, 59)
#     }
# }

# # TODO: Implement CLI options to disable the pygame interface and train the neural net
# # @click.command()
# # @click.option("--train")
# def main():
#     # Pygame init
#     pygame.init()
#     screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
#     pygame.display.set_caption("Neural Net Game")
#     running = True
    
#     global loc_dict
#     loc_dict = {
#         "pet": (20, 20),
#         "food": (10, 10),
#         "water": (30, 10)
#     }
#     sprites = pygame.sprite.Group()
#     pet = Sprites.PetSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["pet"])
#     sprites.add(pet)

#     food = Sprites.GenericDiminishingSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["food"], 5)
#     sprites.add(food)

#     water = Sprites.GenericDiminishingSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["water"], 5)
#     sprites.add(water)

#     active_tool = "food"
#     # Pygame loop
#     while running:
#         screen.fill(SPRITE_DATA["colors"]["background"])

#         # Update the positions of the sprites
#         temp_x, temp_y = coord_converter.gridToScreen(loc_dict["pet"][0], loc_dict["pet"][1], GRID_SIZE)
#         pet.updatePos(temp_x, temp_y)
        
#         temp_x, temp_y = coord_converter.gridToScreen(loc_dict["food"][0], loc_dict["food"][1], GRID_SIZE)
#         food.updatePos(temp_x, temp_y)

#         temp_x, temp_y = coord_converter.gridToScreen(loc_dict["water"][0], loc_dict["water"][1], GRID_SIZE)
#         water.updatePos(temp_x, temp_y)

#         updatePositions()
#         sprites.draw(screen)

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:

#                 if event.key == pygame.K_1:
#                     active_tool = "food"
#                 elif event.key == pygame.K_2:
#                     active_tool = "water"

#                 elif event.key == pygame.K_f:
#                     if active_tool == "food":
#                         food.use()
#                         if food.uses <= 0:
#                             loc_dict["food"] = (-1, -1)
#                     elif active_tool == "water":
#                         water.use()
#                         if water.uses <= 0:
#                             loc_dict["water"] = (-1, -1)

#             elif event.type == pygame.MOUSEBUTTONDOWN:
#                 # Updates the food position to the grid cell that was clicked
#                 # TODO: Allow for clicking to change what is placed - i.e water, food, toys, etc
#                 if active_tool == "food":
#                     food.replenish()
#                 elif active_tool == "water":
#                     water.replenish()
#                 loc_dict[active_tool] = coord_converter.screenToGrid(event.pos[0], event.pos[1], GRID_SIZE)
        
#         # Push the changes to the screen
#         pygame.display.flip()

# def updatePositions():
#     global world
#     world = numpy.zeros(shape = (GRID_X, GRID_Y), dtype = int)
#     world[loc_dict["pet"][0]][loc_dict["pet"][1]] = 1
#     world[loc_dict["food"][0]][loc_dict["food"][1]] = 2
#     world[loc_dict["water"][0]][loc_dict["water"][1]] = 3