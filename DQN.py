import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import gym
import numpy as np
from collections import namedtuple
import random
from matplotlib import pyplot as pl
from IPython.display import clear_output
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN_MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dropout_prob=0,seed = None):
        super().__init__()

        if not seed == None:
            random.seed(seed)
        # original architecture
        self.lin1 = nn.Linear(in_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.lin2 = nn.Linear(hidden_size, out_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        # omri & harel cnn architecture
        # self.conv1 = nn.Conv2d(3,32,kernel_size=3)
        # self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        # self.conv3 = nn.Conv2d(64,64,kernel_size=3)
        # self.fc1 = nn.Linear()



    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.lin1(x)))
        return self.dropout2(self.lin2(x))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        mem_size = len(self.memory)
        batch = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def __len__(self):
        return len(self.memory)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
