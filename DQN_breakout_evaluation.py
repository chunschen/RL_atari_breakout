import gymnasium as gym
import atari_breakout_crop

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