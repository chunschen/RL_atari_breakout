import gymnasium as gym

from stable_baselines3 import PPO
import stable_baselines3.common.atari_wrappers
from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.env_checker import check_env
import atari_breakout_crop

#env = gym.make("CartPole-v1", render_mode="rgb_array")

env = make_vec_env('Breakout-v4', n_envs=4)#, render_mode="human")

#env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
#env = gym.wrappers.FrameStack(env, 4)
#env = atari_breakout_crop.atari_breakout_wrapper(env, 160, 150)
#check_env(env)
model = PPO("CnnPolicy", env, tensorboard_log="F:\RF projects\\runs\\breakout_logs_6", verbose=1)

model.learn(total_timesteps=9000000, log_interval=4)

model.save("F:\RF projects\PPO_breakout6")
env.close()

del model # remove to demonstrate saving and loading


env = gym.make('Breakout-v4', render_mode="human")

#env = gym.make('Breakout-v4')#, render_mode="human")
#env = atari_breakout_crop.atari_breakout_wrapper(env, 160, 150)
#env = stable_baselines3.common.atari_wrappers.AtariWrapper(env)
#env = gym.wrappers.FrameStack(env, 4)

model = PPO.load("F:\RF projects\PPO_breakout6")

obs, info = env.reset()
for i in range(1500):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, terminate, info= env.step(action)
    #env.render("human")
    if done:
        env.reset()
env.close()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()