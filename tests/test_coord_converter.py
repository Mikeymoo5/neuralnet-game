from neuralnetgame.UTIL import coord_converter

def test_screenToGrid():
    grid_size = 16
    assert coord_converter.screenToGrid(0, 0, grid_size) == (0, 0)
    assert coord_converter.screenToGrid(15, 15, grid_size) == (0, 0)
    assert coord_converter.screenToGrid(16, 16, grid_size) == (1, 1)
    assert coord_converter.screenToGrid(17, 16, grid_size) == (1, 1)
    assert coord_converter.screenToGrid(32, 32, grid_size) == (2, 2)
    assert coord_converter.screenToGrid(33, 17, grid_size) == (2, 1)

def test_gridToScreen():
    pass