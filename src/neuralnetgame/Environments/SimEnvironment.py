import gym
from gym import spaces
import numpy as np
import pygame

class SimEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, render_mode = None, cell_size = 16, grid_size = 40):
        
        # Define the action and observation space
        high = 2 # Grass (0), food (1), water (2)
        self.observation_space = spaces.Dict({
            "world": spaces.Box(0, high, shape=(grid_size, grid_size), dtype=int),
            "pet": spaces.Dict({
                "loc": spaces.Box(0, grid_size-1, shape=(2,), dtype=int),
                "hunger": spaces.Box(0, 100, shape=(1,), dtype=int),
                "thirst": spaces.Box(0, 100, shape=(1,), dtype=int),
            })
        })
        # nothing, up, down, left, right, interact - 6 possible actions
        self.action_space = spaces.Discrete(6)
        
        self.cell_size = cell_size
        self.grid_size = grid_size

        self._action_map = {
            0: [0, 0], # Nothing
            1: [0, -1], # Up
            2: [0, 1], # Down
            3: [-1,0], # Left
            4: [1, 0], # Right
            5: "interact" # Interact
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _move_pet(self, move: list):
        new_loc = np.array(self._pet["loc"]) + np.array(move)
        if new_loc < 0 or new_loc >= self.grid_size or new_loc == self._food_loc or new_loc == self._water_loc:
            return {
                "action_type": "move",
                "status": "fail",
                "loc": self._pet["loc"]
            }
        else:
            return {
                "action_type": "move",
                "status": "success",
                "loc": new_loc
            }

    def _interact_pet(self):
        #TODO: re-write to only do required calculations to optimize interactions
        food_dist = np.linalg.norm(np.array(self._pet["loc"]) - np.array(self._food_loc))
        water_dist = np.linalg.norm(np.array(self._pet["loc"]) - np.array(self._water_loc))
        if food_dist < 1:
            self._pet["hunger"] = np.minimum(self._pet["hunger"] + 20, 100)
            return {"action_type": "interact", "status": "success", "interact_type": "food"}
        elif water_dist < 1:
            self._pet["thirst"] = np.minimum(self._pet["thirst"] + 20, 100)
            return {"action_type": "interact", "status": "success", "interact_type": "water"}
        else:
            return {"action_type": "interact", "status": "fail", "interact_type": "none"}
        
    def _get_obs(self):
        return {
            "world": self._world,
            "pet": {
                "loc": self._pet["loc"],
                "hunger": self._pet["hunger"],
                "thirst": self._pet["thirst"]
            }
        }
    
    def _get_info(self):
        return {
            "temp": "temp"
        }
    
    # Calculate the reward
    def _get_reward(self, action):
        reward = 0

        # Punish the agent for dying and immediately return
        if self._pet["hunger"] <= 0 or self._pet["thirst"] <= 0:
            reward = -1000
            return reward
        
        # Reward the agent for eating or drinking when under 80 hunger or thirst
        if action["action_type"] == "interact":
            if action["status"] == "success":
                if action["interact_type"] == "food":
                    if self._pet["hunger"] <= 80:
                        reward += 3
                if action["interact_type"] == "water":
                    if self._pet["thirst"] <= 80:
                        reward += 3

        # Punish the agent for low hunger or thirst
        if self._pet["hunger"] <= 40:
            reward -= 2
        if self._pet["thirst"] <= 40:
            reward -= 2
        
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._world = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._pet = {
            "loc": [self.grid_size//2, (self.grid_size//4)*3],
            "hunger": 100,
            "thirst": 100
        }

        self._food_loc = [self.grid_size//4, self.grid_size//4]
        self._water_loc = [(self.grid_size//4)*3, self.grid_size//4]

        observation = self._get_obs()
        info = self._get_info()
        # Return an observation of the environment - In this case, the world array. Also return auxillary information
        return observation, info
    
    def step(self, action):
        # Convert an action into a dict {type, status, loc/none}
        action = self._action_map[action]
        if action is list:
            self.action = self._move_pet(action)
            self._pet["loc"] = action
        elif action == "interact":
            self.action = self._interact_pet()
        else:
            # Anything that isnt a list and isnt a string is invalid - how did we get here?
            raise ValueError("Invalid action - Achievement unlocked: How did we get here?")
        # Decrease hunger and thirst every step
        self._pet["hunger"] -= 0.75
        self._pet["thirst"] -= 0.75

        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward(action)
        terminated = False

        # Check if the pet is dead
        if self._pet["hunger"] <= 0 or self._pet["thirst"] <= 0:
            terminated = True

        return observation, reward, terminated, info
    
    def _render_frame(self):
        pass #TODO: Render the current state of the environment

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()