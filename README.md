# RL_atari_breakout

## Summary:
This project is to train an agent using reinorcement learning to play the Atari game "breakout". 
The DQN_breaks.py contains the training and performance evaluation of a DQN (deep q-learning network) model.Currently this DQN-based agent can achieve around 50~60 points in average after training the model by playing 5000 games (~12 hours with Nvidia 2060 GPU).

The atari-breakout_crop.py is a wrapper to the environment(Atari game), it preprocessed the image frames and modified the action to reduce delay of the game after game reset.