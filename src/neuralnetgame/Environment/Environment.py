import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimEnvironment(gym.Env):
    def __init__(self, cell_size = 16, grid_size = 40):
        # Define the action and observation space
        high = 2 # Grass (0), food (1), water (2)
        self.observation_space = {
            "world": spaces.Box(0, high, shape=(grid_size, grid_size), dtype=int),
            "pet": {
                "loc": spaces.Box(0, grid_size-1, shape=(2,), dtype=int),
                "hunger": spaces.Box(0, 100, shape=(1,), dtype=int),
                "thirst": spaces.Box(0, 100, shape=(1,), dtype=int)
            }
        }
        # nothing, up, down, left, right, interact - 6 possible actions
        self.action_space = spaces.Discrete(6)
        
        self.cell_size = cell_size
        self.grid_size = grid_size

        self._action_to_str = {
            0: "nothing",
            1: "up",
            2: "down",
            3: "left",
            4: "right",
            5: "interact"
        }
    def _move_pet(self, move: list):
        pass #TODO Accept an array, add it to the pet's location, and check for bounds
    def _interact_pet(self):
        pass #TODO Check if the pet is near a food or water tile, then update stats accordingly
    def _get_obs(self):
        return {
            "world": self._world,
            "pet": {
                "loc": self._pet_loc,
                "hunger": self._pet_hunger,
                "thirst": self._pet_thirst
            }
        }
    
    def _get_info(self):
        return {
            "temp": "temp"
        }
    
    def _render_frame(self):
        pass #TODO: Render the current state of the environment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._world = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._pet_loc = [self.grid_size//2, (self.grid_size//4)*3]
        self._pet_hunger = 100
        self._pet_thirst = 100

        self._food_loc = [self.grid_size//4, self.grid_size//4]
        self._water_loc = [(self.grid_size//4)*3, self.grid_size//4]

        observation = self._get_obs()
        info = self._get_info()
        # Return an observation of the environment - In this case, the world array. Also return auxillary information
        return observation, info
    
    def step(self, action):
        act_str = self._action_to_str[action]

        # TODO: Find a more efficient way to do this
        if act_str == "nothing":
            act_type = 0
            move = np.array([0, 0])
        elif act_str == "up":
            act_type = 0
            move = np.array([0, -1])
        elif act_str == "down":
            act_type = 0
            move = np.array([0, 1])
        elif act_str == "left":
            act_type = 0
            move = np.array([-1, 0])
        elif act_str == "right":
            act_type = 0
            move = np.array([1, 0])
        elif act_str == "interact":
            act_type = 1

        if act_type == 0:
            self._move_pet(move)
        elif act_type == 1:
            self._interact_pet()