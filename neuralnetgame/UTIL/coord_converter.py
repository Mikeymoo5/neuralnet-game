import math
import pygame
# If the screen coords are at the edge of two grid cells, it will return the higher value grid cell
def screenToGrid(x, y, grid_size):
    x_low = math.floor(x / grid_size) * grid_size
    y_low = math.floor(y / grid_size) * grid_size

    grid_x = (x_low / grid_size) 
    grid_y = (y_low / grid_size)

    return (grid_x, grid_y)

def gridToScreen(x, y, grid_size):
    screen_x = x * grid_size + grid_size / 2
    screen_y = y * grid_size + grid_size / 2
    return (screen_x, screen_y)

def gridToRect(x, y, grid_size):
    screen_x, screen_y = gridToScreen(x, y, grid_size)
    return pygame.Rect(screen_x - grid_size / 2, screen_y - grid_size / 2, grid_size, grid_size)
