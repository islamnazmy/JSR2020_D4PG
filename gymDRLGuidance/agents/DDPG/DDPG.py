import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import gymDRLGuidance
import tensorflow as tf

from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import BaseCallback, EvalCallback

def run(showPlots):
    if not showPlots:

        # Create log dir
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create and wrap the environment
        env = gym.make(id="simpleDRLGuidance-v1")
        env = Monitor(env, log_dir)

        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(1) * np.ones(n_actions))

        # Instantiate the agent
        model = DDPG(LnMlpPolicy, env, verbose=1, param_noise=param_noise,
                     action_noise=action_noise, actor_lr=1e-4,
                     critic_lr=1e-4, batch_size=256, buffer_size=1e6, tensorboard_log=log_dir,
                     nb_train_steps=450, nb_rollout_steps=450, nb_eval_steps=450)
        # Create the callback: check every 1000 steps
        eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                     log_path=log_dir, n_eval_episodes=50, eval_freq=1000,
                                     deterministic=True, render=False)
        # Train the agent
        model.learn(total_timesteps=1e5, callback=eval_callback)
        # Save the agent
        model.save("ddpg_DRLGuidance")
        del model  # remove to demonstrate saving and loading

    else:

        # Create and wrap the environment
        env = gym.make(id="simpleDRLGuidance-v1")

        # Load the trained agent
        model = DDPG.load("ddpg_DRLGuidance", env=env)

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)

        print(mean_reward)

        # Enjoy trained agent
        initial_state = env.reset()
        obs = initial_state
        position = env.state
        hold_point_position = env.hold_point
        docking_port_position = env.docking_port
        reward_vec = [0]
        while not env.is_done():
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            position = np.vstack((position, env.state))
            hold_point_position = np.vstack((hold_point_position, env.hold_point))
            docking_port_position = np.vstack((docking_port_position, env.docking_port))
            reward_vec = np.vstack((reward_vec, rewards))

        data = np.load('evaluations.npz')
        training_data = [np.mean(data['results'][i]) for i in range(len(data['timesteps']))]

        plt.figure(1)
        plt.plot(position[:, 0], position[:, 1])
        plt.plot(env.target_location[0], env.target_location[1], 'r')
        plt.plot(hold_point_position[:, 0], hold_point_position[:, 1], 'b')
        plt.plot(docking_port_position[:, 0], docking_port_position[:, 1], 'g')
        plt.plot(position[0, 0], position[0, 1], 'kx')
        plt.figure(2)
        plt.plot(reward_vec)
        plt.figure(3)
        plt.plot(position[:, 2])
        plt.plot(hold_point_position[:, 2], 'r-')
        plt.figure(4)
        plt.plot(data['timesteps'], training_data)
        plt.show()


if __name__ == "__main__":
    run(showPlots=False)
