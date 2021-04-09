import gym
import numpy as np
import matplotlib.pyplot as plt
import gymDRLGuidance

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.env_checker import check_env

env = gym.make(id="simpleDRLGuidance-v1")
check_env(env)

model = DDPG(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ddpg_DRLGuidance")

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_DRLGuidance", env=env)

obs = env.reset()
position = env.state[:2]
while not env.is_done():
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    position = np.vstack((position, env.state[:2]))

plt.plot(position[:, 0], position[:, 1])