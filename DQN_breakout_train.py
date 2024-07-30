import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from DQN_core import DQNAgent
import atari_breakout_crop
import pickle 
import bz2
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
writer = SummaryWriter(f'{dir_path}\\runs\\breakout_logs_3_2')
memory_writer = bz2.open('DQN_memory.obj', 'wb') 


# Initialize environment and agent with Experience Replay Buffer
env = gym.make('Breakout-v4',frameskip = 4)#, render_mode="human")
env = atari_breakout_crop.atari_breakout_wrapper(env, 160, 150)
env = gym.wrappers.FrameStack(env, 4)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim, lr=0.00025, gamma=0.99, epsilon=0.3, epsilon_decay=0.9999, warmup=10000, buffer_size=1000000, update_step=5, update_repeat=50)

# continue the training from previous state, comment out this line if you want to train from the scratch

agent.model.load_state_dict(torch.load(f"{dir_path}\saved_agent_model_3\\retrained-model2_1-1875"))

# Train the DQN agent with Experience Replay Buffer
batch_size = 64
num_episodes = 5000
totalCt = 0
stat_rewards = 0
training_stat_steps = 0


# start training the DQN model for 'num_episodes' episodes. Each episode is one complete game which has five lives
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    replayAvgSum = 0
    replayTargetSum = 0
    ct = 0

    preLive = 5
    loss_live = False
    live_frame_num = 0
    live_frame_start = 0

    nofire_ct = 0

    while not done:
        ct += 1
        action = agent.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        
        episode_frame_number = info['episode_frame_number']

        live = info['lives']
        if live < preLive:
            loss_live = True
            preLive = live
            live_frame_start = episode_frame_number
            reward = -0.5
        else:
            loss_live = False
            live_frame_num = episode_frame_number - live_frame_start

        if type(state) is tuple:
            state = state[0]

        agent.remember(state, action, reward, next_state, done, loss_live)

        state = next_state
        total_reward += reward
        stat_rewards += reward
        totalCt += 1
        training_stat_steps += 1

        replayAvgSum += agent.replay_avg_loss
        replayTargetSum += agent.avg_target
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}\n action count: {ct}, avg Loss in replay: {replayAvgSum/ct}, avg target: {replayTargetSum/ct}")
    
    if (episode > 1) and (episode % agent.update_step == 0):
        for i in range(0, agent.update_repeat):
            agent.replay(batch_size)
        print("** Updating target model")
        agent.update_params(agent.model, agent.target_model)
        print('epsilon: %f' % agent.epsilon)

    if (episode > 1) and (episode % agent.update_step == 0):
        writer.add_scalar("avg training loss",replayAvgSum/ct, totalCt)
        writer.add_scalar("Total Reward",stat_rewards/training_stat_steps, totalCt)
        writer.add_scalar("Avg replay target Q",replayTargetSum/ct, totalCt)
        stat_rewards = 0
        training_stat_steps = 0

    if  (episode > 1) and (episode % 25 == 0):
        torch.save(agent.model.state_dict(), f"{dir_path}\saved_agent_model_3\\retrained-model2_2-{episode}")
        if(episode % 250 == 0):
            pickle.dump(agent.memory, memory_writer)
            memory_writer.flush()
writer.close()
memory_writer.close()