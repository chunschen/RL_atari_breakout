
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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