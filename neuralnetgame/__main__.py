import pygame
import Sprites.Sprites as Sprites
import numpy
# import click
from UTIL import *

# Gymnasium and Environment imports
from Environments.SimEnvironment import SimEnv
import gymnasium as gym

# Tensorflow imports

import os
import sys

learning_rate = 1e-3 # Set the size of steps the optimizer will take

# Misc constants
GRID_SIZE = 16
GRID_X = 40
GRID_Y = 40

SCREEN_WIDTH = GRID_SIZE * GRID_X
SCREEN_HEIGHT = GRID_SIZE * GRID_Y
world = numpy.zeros(shape = (GRID_X, GRID_Y), dtype = int)
SPRITE_DATA = {
    "colors": {
        "pet": (115, 113, 81),
        "food": (92, 62, 43),
        "water": (15, 159, 255),
        "background": (23, 194, 59)
    }
}

# TODO: Implement CLI options to disable the pygame interface and train the neural net
# @click.command()
# @click.option("--train")
def main():
    # A lot of the the following TF code is from the TF Agents guide - https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial - I have added comments using my knowledge of what is happening
    # Environment setup
    train_py_env = suite_gym.load('SimEnv-v0', gym_env_wrappers=[FlattenObservation])
    # eval_py_env = suite_gym.load('SimEnv-v0')
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    # eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (100, 50)
    action_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_spec.maximum - action_spec.minimum + 1 # honestly no clue what this does)
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params] # Create the hidden dense layers

    # Define the last layer of the network - It should have the same number of outputs as the number of actions (6 in this case)
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None, 
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.3, maxval=0.3
        ),
        bias_initializer=tf.keras.initializers.Constant(-0.2)
    ) 

    # Create the Q network by combining the hidden dense layers (dense_layers) and the output layer (q_values_layer)
    # q_net = sequential.Sequential(dense_layers + [q_values_layer])
    q_net = sequential.Sequential([q_values_layer])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network= q_net,
        optimizer= optimizer,
        td_errors_loss_fn= common.element_wise_squared_loss,
        train_step_counter= train_step_counter
    )

    agent.initialize()

    agent.train = common.function(agent.train) # I guess this optimizes it somehow
    agent.train_step_counter.assign(0) # Resets the training step counter
    r = average_return(train_env, agent.policy, 10) # Evaluates the agent for 10 episodes
    print(f'Baseline return: {r}')
    collect_driver = py_driver.PyDriver(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True
        ),
        [rb_observer],
        max_steps=collect_steps_per_iteration)
    agent.train()
    episodes = 10
    for i in range(episodes):
        pass
    

    # END TF AGENTS CODE - TY TF AGENTS GUIDE <3

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Net Game")
    running = True
    
    global loc_dict
    loc_dict = {
        "pet": (20, 20),
        "food": (10, 10),
        "water": (30, 10)
    }
    sprites = pygame.sprite.Group()
    pet = Sprites.PetSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["pet"])
    sprites.add(pet)

    food = Sprites.GenericDiminishingSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["food"], 5)
    sprites.add(food)

    water = Sprites.GenericDiminishingSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["water"], 5)
    sprites.add(water)

    active_tool = "food"
    # Pygame loop
    while running:
        screen.fill(SPRITE_DATA["colors"]["background"])

        # Update the positions of the sprites
        temp_x, temp_y = coord_converter.gridToScreen(loc_dict["pet"][0], loc_dict["pet"][1], GRID_SIZE)
        pet.updatePos(temp_x, temp_y)
        
        temp_x, temp_y = coord_converter.gridToScreen(loc_dict["food"][0], loc_dict["food"][1], GRID_SIZE)
        food.updatePos(temp_x, temp_y)

        temp_x, temp_y = coord_converter.gridToScreen(loc_dict["water"][0], loc_dict["water"][1], GRID_SIZE)
        water.updatePos(temp_x, temp_y)

        updatePositions()
        sprites.draw(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_1:
                    active_tool = "food"
                elif event.key == pygame.K_2:
                    active_tool = "water"

                elif event.key == pygame.K_f:
                    if active_tool == "food":
                        food.use()
                        if food.uses <= 0:
                            loc_dict["food"] = (-1, -1)
                    elif active_tool == "water":
                        water.use()
                        if water.uses <= 0:
                            loc_dict["water"] = (-1, -1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Updates the food position to the grid cell that was clicked
                # TODO: Allow for clicking to change what is placed - i.e water, food, toys, etc
                if active_tool == "food":
                    food.replenish()
                elif active_tool == "water":
                    water.replenish()
                loc_dict[active_tool] = coord_converter.screenToGrid(event.pos[0], event.pos[1], GRID_SIZE)
        
        # Push the changes to the screen
        pygame.display.flip()

def updatePositions():
    global world
    world = numpy.zeros(shape = (GRID_X, GRID_Y), dtype = int)
    world[loc_dict["pet"][0]][loc_dict["pet"][1]] = 1
    world[loc_dict["food"][0]][loc_dict["food"][1]] = 2
    world[loc_dict["water"][0]][loc_dict["water"][1]] = 3

def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'
        )
    )

# Used when evaluating an agent - This is a custom implementation - I think there is a metric function that does this more efficiently
def average_return(env, policy, episodes=10):
    total_return = 0
    for i in range(episodes):
        # Runs each episode
        time_step = env.reset()

        while not time_step.is_last():
            # Runs each step of the episode
            action = policy.action(time_step)
            time_step = env.step(action)
            total_return += time_step.reward
    avg = total_return / episodes
    return avg.numpy()[0] # Returns the average reward, converted from a tensor to a numpy array

            

        



if __name__ == "__main__":
    main()