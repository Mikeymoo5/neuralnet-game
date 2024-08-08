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
import pickle

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

timer = 0

steps_done = 0

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Set the device to use for the neural net
device = torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

click.echo(click.style(f"Using device: {device}", fg="cyan"))

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

# This function will occasionally choose a random action instead of a predicted action, with the odds decreasing as time goes on
# Taken from the Pytorch Reinforcement Learning tutorial
def select_action(state):
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
def train_agent(n_episodes, wait_time=0, render=True, report_interval=10, save_interval=100, resume=False):
    # Run every episode
    for i_episode in range(n_episodes):
        click.echo(click.style(f"RESUME: {resume}", fg="yellow"))
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        replay_memory = Replay(10000)
        # Run every step
        for i in count():
            action = select_action(state) # Selects an action based on the state of the environment
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device) # Converts the reward to a tensor with a single value
            done = terminated or truncated # The episode is done if the environment is terminated or truncated

            # Stop the episode if the environment is terminated
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
            if i % report_interval == 0:
                click.echo(click.style(f"Episode {i_episode} - Step {i} - Reward: {reward.item()}", fg="white"))
            if i_episode % save_interval == 0 & i == 0:
                # Save the replay memory
                rpl = open(f"replays/replay_{i_episode}.rpl", "wb") # Open a file to write the replay memory to, or create a new one if it does not exist
                pickle.dump(replay_memory.memory, rpl) # Write the replay memory to the file
                rpl.close() # Close the file
                replay_memory = Replay(10000) # Reset the replay memory
                if resume:
                    torch.save(policy_net.state_dict(), f"models/model_chk_{i_episode}.pt")
                else:
                    torch.save(policy_net.state_dict(), f"models/model_{i_episode}.pt")
                click.echo(click.style(f"Saving model - Episode {i_episode}", fg="green"))

            # Push the transition to memory 
            memory.push(state, action, next_state, reward)

            # Push the transition to the replay memory and progress to the next state
            replay_memory.push(observation, action, next_state, reward)
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
    if resume:
        torch.save(policy_net.state_dict(), f"models/model_chk_{i_episode}.pt")
    else:
        torch.save(policy_net.state_dict(), f"models/model_{i_episode}.pt")
    plt.ioff()
    plt.show()
def play(model_name="model"):
    global timer
    try:
        # NOTE: Weights_only is set to true so that only the weights are loaded (duh), preventing malicious code execution from unknown model sources
        net = DQN(n_actions, n_observations).to(device)
        net.load_state_dict(torch.load(f"models/{model_name}.pt", map_location=device, weights_only=True))
        # net.eval()
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
    state, info = env.reset()
    while env.pyg_running:
        env.render_frame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if event.button == 1:
                    env.on_mouse_click(x, y, "food")
                elif event.button == 3:
                    env.on_mouse_click(x, y, "water")
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
        delta = env.clock.tick(15) / 1000
        timer += delta
         # Wait for a second before taking the next step
        if timer >= 1:
            timer -= 1
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                actions = net(torch.from_numpy(state).float())
                action = torch.argmax(actions).item()
            observation, reward, terminated, truncated, _ = env.step(action)
            if terminated: env.close()
            state = observation
        
@click.command()

@click.option("--train", is_flag=True)
@click.option("--model-name", type=str, default="pretrained-model")
@click.option("--resume", type=str, default=None)
@click.option("--train-duration", type=int, default=1000)
@click.option("--report-interval", type=int, default=10)
@click.option("--save-interval", type=int, default=100)

@click.option("--replay", is_flag=True)
@click.option("--replay-name", type=str, default=None)

def __main__(train, train_duration, report_interval, save_interval, model_name, resume, replay, replay_name) -> None:
    global device
    global policy_net
    if train:
        click.echo(click.style("Starting in training mode", fg="green"))
        if resume is not None:
            # Attempt to load the model
            try:
                policy_net.load_state_dict(torch.load(f"models/{resume}.pt", map_location=device))
            except FileNotFoundError:
                click.echo(click.style("Model not found. Exiting...", fg="red"))
                sys.exit(1)
            click.echo(click.style("Checkpoint loaded", fg="green"))
        train_agent(train_duration, report_interval=report_interval, save_interval=save_interval, resume=True if resume is not None else False)
        # plotting.plot_durations(episode_durations=episode_durations, is_ipython=is_ipython, show_result=True) #TODO: show plot when arg is passed
    elif replay:
        viewReplay(replay_name)
    else:
        # TODO: Fix the bug that causes Torch DirectML to crash the game
        # if torch_directml.is_available():
        #     device = torch_directml.device(
        #         torch_directml.default_device()
        #     ) 
        click.echo(click.style("Starting in play mode", fg="green"))
        play(model_name)
        
if __name__ == "__main__":
    __main__()