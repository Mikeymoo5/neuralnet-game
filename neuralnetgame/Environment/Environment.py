class Environment:
    def __init__(self):
        pass
    
    # From my limited RL knowledge, this should return the state of the environment after the action has been taken
    # Action should be an integer representing the action the agent wants to take; 0 for up, 1 for down, 2 for left, 3 for right, 4 for interact
    def step(self, action):
        observation = None
        reward = None

        # Return a new observation, as well as a reward
        # The observation should either be the pets line of sight or the entire grid; tbd
        return (observation, reward)