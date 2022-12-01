import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

env = gym.make('CliffWalking-v0', render_mode = "human")
env.reset()
gamma = 0.99
torch.manual_seed(1)


def basisVector(i,size):
    a = np.zeros(size)
    a[i] = 1.0
    return a

class PolicyNet(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size=64):
        super(PolicyNet, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.fc2 = torch.nn.Linear(hidden_layer_size, output_size)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = torch.from_numpy(basisVector(x,self.input_size)).float() # this is for cliff-walking ONLY
        return self.softmax(self.fc2((self.fc1(x))))

    def get_action_and_logp(self, x):
        # print(x)
        action_prob = self.forward(x)
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()
        logp = m.log_prob(action)
        return action.item(), logp

    def act(self, x):
        action, _ = self.get_action_and_logp(x)
        return action

policy = PolicyNet(48,env.action_space.n)
policy.load_state_dict(torch.load("optimalCliffWalkController.pt"))


def select_action(state):
    return policy.act(state)

def main(episodes):
    global env
    for _ in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        state = state[0]
        done = False       

        for _ in range(1000):
            action = select_action(state)
            state, reward, done, truncated, info = env.step(action)

            if done or truncated:
                observation, info = env.reset()


episodes = 10
main(episodes)







