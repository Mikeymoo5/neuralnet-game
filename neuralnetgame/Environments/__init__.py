from gymnasium.envs.registration import register

register(
    id='SimEnv-v0',
    entry_point='Environments.SimEnvironment:SimEnv',
    nondeterministic=True
)