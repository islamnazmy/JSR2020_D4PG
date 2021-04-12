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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                    self.model.save(self.save_path)

        return True


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

        # Custom MLP policy of two layers of size 32 each with tanh activation function
        # policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64, 64])
        # Instantiate the agent
        model = DDPG(LnMlpPolicy, env, verbose=1, param_noise=param_noise,
                     action_noise=action_noise, actor_lr=1e-4,
                     critic_lr=1e-4, batch_size=256, buffer_size=1e6, tensorboard_log=log_dir)
        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                     log_path=log_dir, n_eval_episodes=50, eval_freq=1000,
                                     deterministic=True, render=False)
        # Train the agent
        model.learn(total_timesteps=5e6, callback=eval_callback)
        # Save the agent
        model.save("ddpg_DRLGuidance")
        del model  # remove to demonstrate saving and loading

    else:

        # Create and wrap the environment
        env = gym.make(id="simpleDRLGuidance-v1")

        # Load the trained agent
        model = DDPG.load("best_model", env=env)

        # Evaluate the agent
        # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)

        # Enjoy trained agent
        obs = env.reset()
        position = env.state[:2]
        reward_vec = [0]
        while not env.is_done():
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            position = np.vstack((position, env.state[:2]))
            reward_vec = np.vstack((reward_vec, rewards))

        data = np.load('evaluations.npz')
        training_data = [np.mean(data['results'][i]) for i in range(len(data['timesteps']))]

        plt.figure(1)
        plt.plot(position[:, 0], position[:, 1])
        plt.plot(1.85, 0.6, 'b*')
        plt.plot(1.85, 1.1, 'r*')
        plt.plot(1.85, 1.6, 'g*')
        plt.plot(3.0, 1.0, 'kx')
        plt.figure(2)
        plt.plot(reward_vec)
        plt.figure(3)
        plt.plot(data['timesteps'], training_data)
        plt.show()


if __name__ == "__main__":
    run(showPlots=False)
