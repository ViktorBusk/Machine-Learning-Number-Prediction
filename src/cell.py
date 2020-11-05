import pygame

class Cell:
    def __init__(self, pos, dimensions, color):
        self.pos = pos
        self.dimensions = dimensions
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, ((self.pos), (self.dimensions)))