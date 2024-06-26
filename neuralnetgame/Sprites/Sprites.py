import pygame
class PetSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, color):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill(color)
        self.rect = self.image.get_rect()

        self.rect.x = 16
        self.rect.y = 16
    def updatePos(self, x, y):
        self.rect.x = x
        self.rect.y = y
class FoodSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, color):
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill(color)
        self.rect = self.image.get_rect()

        self.rect.x = 16
        self.rect.y = 16
    def updatePos(self, x, y):
        self.rect.x = x
        self.rect.y = y