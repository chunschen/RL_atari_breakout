import random
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torchvision.transforms.functional
import gymnasium as gym

# A customized wraper for atari breakout game

class atari_breakout_wrapper(gym.ObservationWrapper):
    def __init__(self, env, frame_height, frame_width):
        super(atari_breakout_wrapper, self).__init__(env)
        self.last_live = 5
        self.fire_now = False
        self.frame_height = frame_height
        self.frame_width = frame_width
        #self.reset(seed = 1)


    def observation(self, observation)->np.ndarray:
        obs_transform = np.transpose(observation, [2, 0, 1])
        obs_tensored = torch.tensor(obs_transform, dtype=torch.uint8)
        observation_resized = torchvision.transforms.functional.resized_crop(obs_tensored, 34, 5, self.frame_height, self.frame_width, [84, 84])
        observation_greyscale = torchvision.transforms.functional.rgb_to_grayscale(observation_resized)

        return observation_greyscale.numpy()
    
    def reset(self, seed  = None) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(2)

        #self.showObsImage(self.observation(obs))
        self.fire_now = True
        self.last_live = 5
        return self.observation(obs), info
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        live = info['lives']
        if live < self.last_live:
            self.fire_now = True
            self.last_live = live  # Reset the counter
        
        if self.fire_now:
            pre_obs = observation

            while np.array_equal(pre_obs, observation):
                observation, reward, terminated, truncated, info = self.env.step(1)
                if terminated:
                    break
            self.fire_now = False

        new_obs =  self.observation(observation)
        #if reward > 0:
        #   plt.imshow(new_obs[0], cmap='gray')
        return new_obs, reward, terminated, truncated, info
    
    def show_obs_image(self, obs):
        plt.imshow(obs, cmap='gray')