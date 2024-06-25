import pygame
class PetSprite(pygame.sprite.Sprite):
    def __init__(self, width, height):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill("red")
        self.rect = self.image.get_rect()

        self.rect.x = 16
        self.rect.y = 16