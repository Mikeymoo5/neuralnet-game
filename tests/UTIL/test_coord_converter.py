from neuralnetgame.UTIL import coord_converter
GRID_SIZE = 16

def test_screenToGrid():
    assert coord_converter.screenToGrid(0, 0, GRID_SIZE) == (0, 0)
    assert coord_converter.screenToGrid(15, 15, GRID_SIZE) == (0, 0)
    assert coord_converter.screenToGrid(16, 16, GRID_SIZE) == (1, 1)
    assert coord_converter.screenToGrid(17, 16, GRID_SIZE) == (1, 1)
    assert coord_converter.screenToGrid(32, 32, GRID_SIZE) == (2, 2)
    assert coord_converter.screenToGrid(33, 17, GRID_SIZE) == (2, 1)

def test_gridToScreen():
    assert coord_converter.gridToScreen(1, 1, GRID_SIZE) == (24, 24)
    assert coord_converter.gridToScreen(2, 2, GRID_SIZE) == (40, 40)
    assert coord_converter.gridToScreen(3, 2, GRID_SIZE) == (56, 40)
    assert coord_converter.gridToScreen(1, 2, GRID_SIZE) == (24, 40)
    assert coord_converter.gridToScreen(0, 0, GRID_SIZE) == (8, 8)
    assert coord_converter.gridToScreen(0, 1, GRID_SIZE) == (8, 24)