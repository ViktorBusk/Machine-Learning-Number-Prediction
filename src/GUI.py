import pygame
import math
import numpy as np
from numpy import interp

from paint_brush import PaintBrush
from canvas import Canvas
from colors import BLACK, WHITE, RED, GREEN, YELLOW, CUSTOM

class GUI:
    def __init__(self, window_scale_factor):
        # Initialize the pygame library
        pygame.init()
        pygame.font.init() 

        # Set up the drawing window
        self.window_size = ([window_scale_factor[0]*28, window_scale_factor[1]*28])
        pygame.display.set_caption('AI Number Prediction')
        self.screen = pygame.display.set_mode(self.window_size)
        self.running = True
        
        # Set up external objects
        self.paint_brush = PaintBrush()
        self.canvas = Canvas(self.window_size, (28, 28))
        self.prediction_text = ["Prediction: ", "", "Certainty: ", ""]
        self.predicted_prob = 0

    def __poll_events(self):
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        
        # Clear the screen if the user clicks right button
        if pygame.mouse.get_pressed()[2]:
            self.canvas.clear()

    def __map(self, value, start_1, stop_1, start_2, stop_2):
        return interp(value,[start_1, stop_1],[start_2, stop_2])
    
    def __lerp_RGB(self, a, b, t):
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )

    def __get_prediction_color(self, predicted_prob):
        '''
        Linear interpolates between RED, YELLOW and GREEN
        '''
        if predicted_prob < 0.5:
            t = self.__map(predicted_prob, 0, 0.5, 0, 1)
            return self.__lerp_RGB(RED, YELLOW, t)
        else:
            t = self.__map(predicted_prob, 0.5, 1, 0, 1)
            return self.__lerp_RGB(YELLOW, GREEN, t)

    def __on_paint_brush_usage(self):
        if self.paint_brush.using:
            # Get mouse grid position
            grid_pos_x = math.floor(self.paint_brush.pos[0] / self.canvas.cell_dimensions[0])
            grid_pos_y = math.floor(self.paint_brush.pos[1] / self.canvas.cell_dimensions[1])

            # Paint with the brush
            for x in range(grid_pos_x - self.paint_brush.size[0], grid_pos_x + self.paint_brush.size[1]):
                for y in range(grid_pos_y - self.paint_brush.size[2], grid_pos_y + self.paint_brush.size[3]):
                    if x >= 0 and y >= 0 and x < self.canvas.grid_w and y < self.canvas.grid_h:
                        self.canvas.grid[x][y].color = self.paint_brush.color
                        self.canvas.grid[x][y].draw(self.canvas.surface)
                        self.canvas.np_arry[y][x] = 1

    def reset_prediction_output(self):
        self.prediction_text[1] = ""
        self.prediction_text[3] = ""

    def __draw_text(self):
        font_size_1 = 13
        font_size_2 = 16
        text_pos = (15, 15)

        # Load font
        myfont_1 = pygame.font.Font(pygame.font.get_default_font(), font_size_1)
        myfont_2 = pygame.font.Font(pygame.font.get_default_font(), font_size_2)

        # Row 1
        textsurface = myfont_1.render(self.prediction_text[0], True, WHITE, BLACK)
        self.screen.blit(textsurface, text_pos)

        x_offset = text_pos[0] + textsurface.get_width()
        textsurface = myfont_2.render(self.prediction_text[1], True, CUSTOM, BLACK)
        self.screen.blit(textsurface, (text_pos[0] + x_offset, text_pos[1]))
        
        # Row 2
        textsurface = myfont_1.render(self.prediction_text[2], True, WHITE, BLACK)
        self.screen.blit(textsurface, (text_pos[0], text_pos[1] + font_size_2))

        x_offset = text_pos[0] + textsurface.get_width()

        textsurface = myfont_2.render(self.prediction_text[3], True, self.__get_prediction_color(self.predicted_prob), BLACK)
        self.screen.blit(textsurface, (text_pos[0] + x_offset, text_pos[1] + font_size_2))
    
    def __draw(self):
        self.__on_paint_brush_usage()
        self.canvas.draw(self.screen)
        
        self.__draw_text()

        # Flip the display
        pygame.display.flip()

    def __update(self):
        self.paint_brush.update()

    def render(self):
        while(self.running):
            self.__poll_events()
            self.__update()
            self.__draw()
            
        # Done! Time to quit.
        pygame.quit()