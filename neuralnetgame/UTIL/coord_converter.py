import math
def screenToGrid(x, y, grid_size):
    x_low = math.floor(x / grid_size) * grid_size
    x_high = x_low + grid_size

    y_low = math.floor(y / grid_size) * grid_size
    y_high = y_low + grid_size
    
    grid_x = (x_low / grid_size) 
    grid_y = (y_low / grid_size)

    return (grid_x, grid_y)

def gridToScreen(x, y, grid_size):
    low_x = x - (x % grid_size)
    low_y = y - (y % grid_size)
    # high = low + grid_size
    return (low_x/grid_size, low_y/grid_size)