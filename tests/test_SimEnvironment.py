from neuralnetgame.Environments.SimEnvironment import SimEnv
env = SimEnv()
def hunger(x,y):
    env.reset()
    assert env._pet["hunger"] == 100
    env._pet["hunger"] = 80
    assert env._pet["hunger"] == 80
    assert env._food_loc == [5, 5]
    env._pet["loc"] = [x, y]
    env.step(5)
    return env._pet["hunger"]

def thirst(x,y):
    env.reset()
    assert env._pet["thirst"] == 100
    env._pet["thirst"] = 80
    assert env._pet["thirst"] == 80
    assert env._water_loc == [15, 5]
    env._pet["loc"] = [x, y]
    env.step(5)
    return env._pet["thirst"]

def test_food_interact():
    assert hunger(6,5) == 99.5
    assert hunger(5,6) == 99.5
    assert hunger(4, 5) == 99.5
    assert hunger(5, 4) == 99.5

def test_water_interact():
    assert thirst(16,5) == 99
    assert thirst(15,6) == 99
    assert thirst(14, 5) == 99
    assert thirst(15, 4) == 99