import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((420, 640))
    pygame.display.set_caption("Neural Net Game")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
if __name__ == "__main__":
    main()