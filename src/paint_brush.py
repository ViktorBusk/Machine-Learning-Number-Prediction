import pygame
from colors import WHITE

class PaintBrush:
    def __init__(self):
        self.using = False
        self.pos = pygame.mouse.get_pos()
        self.color = WHITE
        self.size = (0, 2, 0, 2)

    def __check_using(self):
        if pygame.mouse.get_pressed()[0]:
            self.using = True
        else:
            self.using = False

    def __update_pos(self):
        self.pos = pygame.mouse.get_pos()

    def update(self):
        self.__update_pos()
        self.__check_using()