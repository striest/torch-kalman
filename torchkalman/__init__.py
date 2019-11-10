from gym.envs.registration import register

register(
	id='NActCartPole-v0',
	entry_point='drivingenvs.envs:NActCartPoleEnv',
	max_episode_steps=500,
)
