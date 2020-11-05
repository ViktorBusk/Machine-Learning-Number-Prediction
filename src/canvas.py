import pygame
import numpy as np
from cell import Cell
from colors import BLACK

class Canvas:
    def __init__(self, size, grid_size):
        self.size = size
        self.surface = pygame.Surface(self.size)
        self.color = BLACK    
        self.grid_w, self.grid_h = grid_size
        self.cell_dimensions = (self.size[0]/self.grid_w, self.size[1]/self.grid_h)
        self.np_arry = np.zeros([self.grid_w, self.grid_h]) 
        
        # Init grid
        self.grid = []
        self.__init_grid()
    
    def __init_grid(self):
        for x in range(self.grid_w):
            temp = []
            for y in range(self.grid_h):
                temp.append(Cell((x*self.cell_dimensions[0], y*self.cell_dimensions[1]), self.cell_dimensions, self.color))
            self.grid.append(temp)

    def draw(self, screen):
        screen.blit(self.surface, (0, 0))

    def clear(self):
        # Clear the surface with a color
        self.surface.fill(self.color)

        # Clear the cells in grid
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                self.grid[x][y].color = self.color

        # Re-init numpy array
        self.np_arry = np.zeros([self.grid_w, self.grid_h], dtype='f') 