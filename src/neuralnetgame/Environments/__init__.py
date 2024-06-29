from gym.envs.registration import register

print("Register init")
register(
    id='SimEnv-v0',
    entry_point='Environments.SimEnvironment:SimEnv',
    max_episode_steps=1000,
    nondeterministic=True
)