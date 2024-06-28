import pygame
import Sprites.Sprites as Sprites
import numpy
# import click
from UTIL import *

# Machine learning *magic*
from Environments.SimEnvironment import SimEnv
import gymnasium as gym

env = gym.make('SimEnv-v0')

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
    
if __name__ == "__main__":
    main()