import pygame
import Sprites.Sprites as Sprites
import numpy
from UTIL import *
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
        "background": (23, 194, 59)
    }
}
def main():
    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neural Net Game")
    running = True
    
    global loc_dict
    loc_dict = {
        "pet_pos": (20, 30),
        "food_pos": (10, 10)
    }
    pet = Sprites.PetSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["pet"])
    pet_sprites = pygame.sprite.GroupSingle()
    pet_sprites.add(pet)

    food = Sprites.FoodSprite(GRID_SIZE, GRID_SIZE, SPRITE_DATA["colors"]["food"])
    food_sprites = pygame.sprite.GroupSingle()
    food_sprites.add(food)

    # Pygame loop
    while running:
        screen.fill(SPRITE_DATA["colors"]["background"])

        temp_x, temp_y = coord_converter.gridToScreen(loc_dict["pet_pos"][0], loc_dict["pet_pos"][1], GRID_SIZE)
        pet.updatePos(temp_x, temp_y)
        
        temp_x, temp_y = coord_converter.gridToScreen(loc_dict["food_pos"][0], loc_dict["food_pos"][1], GRID_SIZE)
        food.updatePos(temp_x, temp_y)

        updatePositions()
        pet_sprites.draw(screen)
        food_sprites.draw(screen)

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Updates the food position to the grid cell that was clicked
                # TODO: Allow for clicking to change what is placed - i.e water, food, toys, etc
                loc_dict["food_pos"] = coord_converter.screenToGrid(event.pos[0], event.pos[1], GRID_SIZE)

def updatePositions():
    global world
    world = numpy.zeros(shape = (GRID_X, GRID_Y), dtype = int)
    world[loc_dict["pet_pos"][0]][loc_dict["pet_pos"][1]] = 1
    world[loc_dict["food_pos"][0]][loc_dict["food_pos"][1]] = 2
    
if __name__ == "__main__":
    main()