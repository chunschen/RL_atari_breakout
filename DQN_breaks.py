import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import math
from gymnasium.wrappers import AtariPreprocessing
import atari_breakout_crop
import pickle 
import bz2

is_train = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('F:\RF projects\\runs\\breakout_logs_3_2')
memory_writer = bz2.open('DQN_memory.obj', 'wb') 

# Define Deep Q-Network (DQN) Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(12544, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim)


    def forward(self, x):
        #x = torch.transpose(x, 0, 2)
        x = x/255  # normalize the input
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = torch.flatten(x, 0)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Define DQN Agent with Experience Replay Buffer
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, warmup, buffer_size, update_step, update_repeat):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.update_step = update_step
        self.update_repeat = update_repeat
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.warmup = warmup

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=0.005)
        self.model.to(device)
        self.target_model.to(device)

        self.replay_avg_loss = 0
        self.avg_target = 0

        for params in self.target_model.parameters():
            params.requires_grad = False

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        
        if type(state) is tuple:
                state = state[0]
        #state = np.transpose(state, (0,3,1,2))
        state_transposed = torch.tensor(state, dtype=torch.float32, device = device)

        q_values = self.model(state_transposed)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done, loss_live):
        self.memory.append((state, action, reward, next_state, done, loss_live))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        self.replay_avg_loss = 0
        loss_sum = 0
        target_sum = 0
        for state, action, reward, next_state, done, loss_live in minibatch:
            target = reward
            if not done:
                #next_state = np.transpose(next_state, (0,3,1,2))
                next_state_transposed = torch.tensor(next_state, dtype=torch.float32, device = device)
                q_targetState = self.target_model(next_state_transposed)
                target = reward + self.gamma * torch.max(q_targetState).item()
            if loss_live:
                target = reward
            if type(state) is tuple:
                state = state[0]
            #state = np.transpose(state, (0,3,1,2))
            state_transposed = torch.tensor(state, dtype=torch.float32, device = device)

            q_curState = self.model(state_transposed)

            target_State = torch.clone(q_curState)

            target_f = torch.flatten(target_State)
            target_f[action] = target
            target_sum += target

            #loss = nn.MSELoss()(torch.tensor(target_f), self.model(torch.tensor(state, dtype=torch.float32)))
            #q_curState = self.model(state_transposed)
            loss = nn.MSELoss()(target_f, q_curState)
            loss_sum += loss.item()
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > 0.01:
            self.epsilon -= (self.epsilon_decay/self.warmup)

        #print("LostSum: in memory replay",lossSum)
        self.replay_avg_loss = loss_sum / batch_size
        self.avg_target = target_sum / batch_size

    def update_params(self, q_m, q_target):
        q_target.load_state_dict(q_m.state_dict())

# Initialize environment and agent with Experience Replay Buffer
if is_train: 
    env = gym.make('Breakout-v4',frameskip = 4)#, render_mode="human")
    #env = AtariPreprocessing(env)
    #env = gym.wrappers.GrayScaleObservation(env)
    env = atari_breakout_crop.atari_breakout_wrapper(env, 160, 150)
    env = gym.wrappers.FrameStack(env, 4)

    #video = RecordVideo(env,"breaks_training_videos")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=0.00025, gamma=0.99, epsilon=0.3, epsilon_decay=0.9999, warmup=10000, buffer_size=1000000, update_step=5, update_repeat=50)
    
    # continue training from previous state
    agent.model.load_state_dict(torch.load(f"F:\RF projects\saved_agent_model_3\\retrained-model2_1-1875"))
    
    # Train the DQN agent with Experience Replay Buffer
    batch_size = 64
    num_episodes = 5000
    totalCt = 0
    stat_rewards = 0
    training_stat_steps = 0
   
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
                #if live_frame_num > 120:
                #    reward += 1/(50 + 750/(live_frame_num/16))

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
            torch.save(agent.model.state_dict(), f"F:\RF projects\saved_agent_model_3\\retrained-model2_2-{episode}")
            if(episode % 250 == 0):
                pickle.dump(agent.memory, memory_writer)
                memory_writer.flush()
    writer.close()
    memory_writer.close()

else:
    env = gym.make('Breakout-v4',frameskip=4, render_mode="human")
    env = atari_breakout_crop.atari_breakout_wrapper(env, 160, 150)
    env = gym.wrappers.FrameStack(env,4)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.001, epsilon_decay=0.999, warmup=1000, buffer_size=10000, update_step=10, update_repeat=50)

    #agent.model = agent.model.load_state_dict(torch.load(f"F:\RF projects\saved_agent_model_2\model-1801"))
    agent.model.load_state_dict(torch.load(f"F:\RF projects\saved_agent_model_3\\retrained-model2_2-1400"))
    # Evaluate the trained agent
    total_rewards = []
    num_episodes_eval = 10
    for _ in range(num_episodes_eval):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
    print(f"Average Total Reward (Evaluation): {np.mean(total_rewards)}")