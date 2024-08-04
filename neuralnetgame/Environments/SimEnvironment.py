import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from UTIL import *

class SimEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode = None, cell_size = 16, grid_size = 20):
        
        # Define the action and observation space
        high = 2 # Grass (0), food (1), water (2)
        # "hunger": spaces.Box(0, 100, shape=(1,), dtype=int),
        # "thirst": spaces.Box(0, 100, shape=(1,), dtype=int),
        self.observation_space = spaces.Dict({
            "world": spaces.Box(0, high, shape=(grid_size, grid_size), dtype=int),
            "pet": spaces.Dict({
                "loc": spaces.Box(0, grid_size-1, shape=(2,), dtype=int),
                "hunger": spaces.Box(low=np.array([-1]), high=np.array([1])),
                "thirst": spaces.Box(low=np.array([-1]), high=np.array([1]))
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
        self.screen = None
        self.clock = None
        self.render_mode = render_mode
        if render_mode == "human":
            self.pyg_running = True

    def _move_pet(self, move: list):
        new_loc = np.array(self._pet["loc"]) + np.array(move)
        new_loc = new_loc.tolist()
        # TODO: Find a less resource intensive way to check for bound issues
        if new_loc[0] < 0 or new_loc[1] < 0 or new_loc[0] >= self.grid_size or new_loc[1] >= self.grid_size or new_loc == self._food_loc or new_loc == self._water_loc:
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
        if food_dist == 1:
            self._pet["hunger"] = np.minimum(self._pet["hunger"] + 20, 100)
            return {"action_type": "interact", "status": "success", "interact_type": "food"}
        elif water_dist == 1:
            self._pet["thirst"] = np.minimum(self._pet["thirst"] + 20, 100)
            return {"action_type": "interact", "status": "success", "interact_type": "water"}
        else:
            return {"action_type": "interact", "status": "fail", "interact_type": "none"}
        
    def _get_obs(self):
        # return {
        #     "world": self._world,
        #     "pet": {
        #         "loc": np.array(self._pet["loc"]),
        #         "hunger": np.array([self._pet["hunger"]], dtype=np.float32),
        #         "thirst": np.array([self._pet["thirst"]], dtype=np.float32)
        #     }
        # }
        return {
            "world": self._world,
            "pet": {
                "loc": np.array(self._pet["loc"]),
                "hunger": np.array([self._pet["hunger"]/100], dtype=np.float32),
                "thirst": np.array([self._pet["thirst"]/100], dtype=np.float32)
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
        # Reward the agent for moving
        if self.action["action_type"] == "move":
            reward += 0.08
        # Reward the agent for eating or drinking when under 80 hunger or thirst
        if self.action["action_type"] == "interact":
            if self.action["status"] == "success":
                if self.action["interact_type"] == "food":
                    if self._pet["hunger"] <= 80:
                        reward += 2
                    else:
                        reward += 2
                if self.action["interact_type"] == "water":
                    if self._pet["thirst"] <= 80:
                        reward += 2
                    else:
                        reward += 2

        # Punish the agent for low hunger or thirst
        if self._pet["hunger"] <= 40:
            food_dist = np.linalg.norm(np.array(self._pet["loc"]) - np.array(self._food_loc))
            if food_dist < 5:
                reward += (0-food_dist) * 0.01
            reward -= 0.1
        if self._pet["thirst"] <= 40:
            water_dist = np.linalg.norm(np.array(self._pet["loc"]) - np.array(self._food_loc))
            if water_dist < 5:
                reward += (0-water_dist) * 0.01
            reward -= 0.1
        
        reward += 0.01
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._world = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self._pet = {
            "loc": [self.grid_size//2, (self.grid_size//4)*3],
            "hunger": 100.0,
            "thirst": 100.0
        }

        self._food_loc = [self.grid_size//4, self.grid_size//4]
        self._water_loc = [(self.grid_size//4)*3, self.grid_size//4]

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # Return an observation of the environment - In this case, the world array. Also return auxillary information
        return observation, info
    
    def step(self, action):
        # Convert an action into a dict {type, status, loc/none}
        action = self._action_map[action]
        if isinstance(action, list):
            self.action = self._move_pet(action)
            self._pet["loc"] = self.action["loc"]
        elif action == "interact":
            self.action = self._interact_pet()
        else:
            # Anything that isnt a list and isnt a string is invalid - how did we get here?
            raise ValueError(f"Invalid action: {action} - Achievement unlocked: How did we get here?")
        # Decrease hunger and thirst every step
        self._pet["hunger"] -= 0.5
        self._pet["thirst"] -= 1

        observation = self._get_obs()
        info = self._get_info()
        reward = self._get_reward(action)
        terminated = False

        # Check if the pet is dead
        if self._pet["hunger"] <= 0 or self._pet["thirst"] <= 0:
            terminated = True

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _render_frame(self):
        if self.screen == None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        if self.clock == None:
            self.clock = pygame.time.Clock()
        self.screen.fill((0,255,0))

        pygame.draw.rect(self.screen, (0,238,255), coord_converter.gridToRect(self._water_loc[0], self._water_loc[1], self.cell_size)) # Draws water
        pygame.draw.rect(self.screen, (255,140,0), coord_converter.gridToRect(self._food_loc[0], self._food_loc[1], self.cell_size)) # Draws food
        pygame.draw.rect(self.screen, (99, 58, 39), coord_converter.gridToRect(self._pet["loc"][0], self._pet["loc"][1], self.cell_size)) # Draws the pet
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()