import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 420))
    pygame.display.set_caption("Neural Net Game")
    running = True
    screen.fill("green")
    while running:

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
if __name__ == "__main__":
    main()