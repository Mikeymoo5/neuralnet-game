from gymnasium.envs.registration import register

register(
    id='SimEnv-v0',
    entry_point='neuralnetgame:Environments',
    max_episode_steps=1000,
    nondeterministic=True
)