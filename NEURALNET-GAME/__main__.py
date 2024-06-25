import pygame
import Sprites.Sprites as Sprites

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

    world = [[0 for x in range(GRID_X)] for y in range(GRID_Y)] # 2D array of 0s
    loc_dict = dict(player_pos = (0, 0), pet_pos = (0, 0))
    pet = Sprites.PetSprite(GRID_SIZE, GRID_SIZE)
    pet_sprites = pygame.sprite.GroupSingle()
    pet_sprites.add(pet)

    # Pygame loop
    while running:

        player_sprites.draw(screen)
        pet_sprites.draw(screen)

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
if __name__ == "__main__":
    main()