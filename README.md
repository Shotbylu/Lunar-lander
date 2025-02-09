# Lunar Lander AI Model

This repository contains an implementation of a Deep Q-Network (DQN) to solve the Lunar Lander environment from OpenAI Gym using reinforcement learning. The goal is to train an agent to autonomously land a lunar module on the moon's surface, optimizing for fuel efficiency and landing safety.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Demo Video](#demo-video)
- [Challenges and Future Work](#challenges-and-future-work)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Lunar Lander environment is a well-known problem in reinforcement learning where an agent must learn to control a lander to safely touch down on the surface of the moon. This project utilizes a Deep Q-Network (DQN) to train the agent, leveraging PyTorch for the neural network implementation and Gymnasium (a fork of OpenAI Gym) for the simulation environment.


## Using Google Colab

You can run this entire project directly in Google Colab without needing to install anything on your local machine. Follow these steps:

1. **Open Google Colab:**
   - Go to [Google Colab](https://colab.research.google.com/).

2. **Upload the Notebook:**
   - Click on "File" > "Upload notebook" and select the `Lunar_Landing.ipynb` file from this repository.

3. **Run the Cells:**
   - Run each cell in the notebook sequentially to set up the environment, train the model, and visualize the results.

4. **Video Demo:**
   - The notebook includes code to generate and display a video of the trained agent in action.

## Dependencies

If you prefer to run the project locally, the following libraries are required:

- **Gymnasium:** A toolkit for developing and comparing reinforcement learning algorithms.
  - Installation: `pip install gymnasium`
- **PyTorch:** An open-source machine learning library for Python, primarily developed by Facebook's AI Research lab.
  - Installation: `pip install torch`
- **NumPy:** A fundamental package for scientific computing in Python.
  - Installation: `pip install numpy`
- **Imageio:** A Python library that provides an easy interface to read and write images and videos.
  - Installation: `pip install imageio`
- **IPython:** A command shell for interactive computing in Python, supporting Jupyter notebooks.
  - Installation: `pip install ipython`

Additional dependencies for local installations:
- **Swig:** Needed for some environments in Gymnasium.
  - Installation: `!apt-get install -y swig`

If using Google Colab, these dependencies are automatically handled by the notebook.

## Approach

The project uses a **Deep Q-Network (DQN)**, which is a reinforcement learning algorithm where a neural network is trained to approximate the Q-value function—a measure of the expected future rewards for taking a given action in a given state.

### Key Concepts Used:
- **Q-Learning:** An off-policy algorithm used to learn the value of an action in a particular state.
- **Replay Memory:** A buffer that stores experiences for the agent to learn from.
- **Epsilon-Greedy Policy:** A strategy used to balance exploration and exploitation by the agent during training.

## Model Architecture

The neural network used in this project consists of:

- **Input Layer:** Takes the state of the environment as input.
- **Hidden Layers:** Two fully connected layers with 64 neurons each, activated by ReLU functions.
- **Output Layer:** Outputs a Q-value for each possible action.

```python
class Network(nn.Module):

  def __init__(self, state_size, action_size, seed=42):
      super(Network, self).__init__()
      self.seed = torch.manual_seed(seed)
      self.fc1 = nn.Linear(state_size, 64)
      self.fc2 = nn.Linear(64, 64)
      self.fc3 = nn.Linear(64, action_size)

  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return self.fc3(x)



-------------------------------------------------------------------------------------------------------------------------------
Training Process
The training process is as follows:

Environment Setup: The Lunar Lander environment is initialized using Gymnasium.
Experience Collection: The agent interacts with the environment, and experiences (state, action, reward, next state, done) are stored in replay memory.
Learning: At each step, the agent samples a batch of experiences from the replay memory to update the Q-network. The network's weights are updated using the Mean Squared Error (MSE) loss function.
Target Network: A target network is used for stabilizing training, and it is updated using a soft update approach.
Evaluation: The agent's performance is evaluated based on the average score over 100 episodes.
Results
The model was trained over 698 episodes, achieving an average score of 200.44, thereby solving the Lunar Lander environment. Here are some results from the training process:

Episode 100: Average Score = -171.75
Episode 300: Average Score = -62.07
Episode 500: Average Score = 20.34
Episode 698: Average Score = 200.44
The environment was considered solved after 598 episodes.

Demo Video
Here is a video demonstration of the trained agent successfully landing the lunar module:


Challenges and Future Work
Challenges:
Exploration vs. Exploitation: Balancing these two during training was challenging, particularly with the sparse rewards in the environment.
Hyperparameter Tuning: Finding the right combination of learning rate, discount factor, and replay buffer size required significant experimentation.
Future Work:
Double DQN: Implementing a Double DQN to address the overestimation bias in Q-learning.
Dueling DQN: This would help the agent to differentiate between the state values and advantages for each action, potentially leading to faster convergence.
Experiment with Other Algorithms: Exploring other reinforcement learning algorithms like A3C or PPO for this task.
