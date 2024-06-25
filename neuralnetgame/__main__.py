import pygame
import Sprites.Sprites as Sprites
import numpy
from UTIL import *
GRID_SIZE = 16
GRID_X = 40
GRID_Y = 40

SCREEN_WIDTH = GRID_SIZE * GRID_X
SCREEN_HEIGHT = GRID_SIZE * GRID_Y

def main():
    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Net Game")
    running = True
    screen.fill("green")
    global loc_dict
    global world
    world = numpy.ndarray(shape = (GRID_X, GRID_Y), dtype = int)
    loc_dict = dict(pet_pos = (30, 30))
    pet = Sprites.PetSprite(GRID_SIZE, GRID_SIZE)
    pet_sprites = pygame.sprite.GroupSingle()
    pet_sprites.add(pet)

    # Pygame loop
    while running:
        pet.updatePos(coord_converter.gridToScreen(loc_dict["pet_pos"][0], loc_dict["pet_pos"][1], GRID_SIZE))
        

        updatePositions()
        pet_sprites.draw(screen)

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

def updatePositions():
    world = [[0 for x in range(GRID_X)] for y in range(GRID_Y)]
    world[loc_dict["pet_pos"][0]][loc_dict["pet_pos"][1]] = 1
    


if __name__ == "__main__":
    main()