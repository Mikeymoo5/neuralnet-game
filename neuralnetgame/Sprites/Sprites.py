import pygame
import numpy
# from UTILS import coord_converter
class GenericSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, color):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((width, height))
        self.default_color = color
        self.color = self.default_color
        self.image.fill(self.default_color)
        self.rect = self.image.get_rect()

        self.rect.x = 16
        self.rect.y = 16

    def updatePos(self, x, y):
        self.rect.x = x
        self.rect.y = y

    def updateColor(self, color):
        self.image.fill(color)

# TODO: Change to inherit GenericSprite
class GenericDiminishingSprite(GenericSprite):
    def __init__(self, width, height, color, max_uses):
        super().__init__(width, height, color)
        self.max_uses = max_uses
        self.uses = max_uses
        self.updateColor(color)
    def use(self):
        self.uses -= 1
        self.color = numpy.multiply(self.color, 0.95)
        self.updateColor(self.color)
    def replenish(self):
        self.uses = self.max_uses
        self.updateColor(self.default_color)

class PetSprite(GenericSprite):
    def __init__(self, width, height, color):
        super().__init__(width, height, color)
