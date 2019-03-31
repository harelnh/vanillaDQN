import torch
import torch.nn as nn
import torch.autograd as autograd


from torch import optim
import torch.nn.functional as F
import gym
import numpy as np
from collections import namedtuple
import random
from matplotlib import pyplot as pl
from IPython.display import clear_output
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DRQN_atary(nn.Module):
    def __init__(self, input_shape, out_size, inner_linear_dim,hidden_dim,lstm_layers,batch=32,traj_len = 100,
                 dropout_prob=0, seed = None, device = torch.device('cpu')):
        super().__init__()

        if not seed == None:
            random.seed(seed)
        # original architecture
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.batch = batch
        self.traj_len = traj_len
        self.lstm_layers = lstm_layers
        self.device = device
        self.hidden = self.init_hidden()
        self.batch_hidden = self.init_batch_hidden()

        # self.lin1 = nn.Linear(in_size, inner_linear_dim)
        # self.dropout1 = nn.Dropout(dropout_prob)
        # self.lstm_layer = nn.LSTM(inner_linear_dim, hidden_dim, lstm_layers,batch_first=True, dropout=dropout_prob,bidirectional=False)
        # self.lin2 = nn.Linear(hidden_dim, out_size)

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), inner_linear_dim),
            nn.ReLU(),
            # nn.Linear(512, self.num_actions)
        )
        self.lstm_layer = nn.LSTM(inner_linear_dim, hidden_dim, lstm_layers, batch_first=True, dropout=dropout_prob,
                                  bidirectional=False)
        self.lin2 = nn.Linear(hidden_dim, out_size)
        # omri & harel cnn architecture
        # self.conv1 = nn.Conv2d(3,32,kernel_size=3)
        # self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        # self.conv3 = nn.Conv2d(64,64,kernel_size=3)
        # self.fc1 = nn.Linear()

    def forward_batch(self,batch_state):
        batch_Q = torch.tensor([]).to(self.device)
        for state in batch_state:
            # state = Variable(torch.FloatTensor(np.float32(state)).to(self.device))
            state = torch.unsqueeze(state,0)
            batch_Q = torch.cat((batch_Q, self.forward(state)))
        return batch_Q

    def forward(self, x):
        is_batch = True if len(x.shape) == 5 else False

        if is_batch:
            x = x.view(self.batch * self.traj_len, 1, x.shape[-1], x.shape[-1])
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            # x = x.view(x.size(0),1, -1)
            x = x.view(self.batch, self.traj_len, -1)
            output, self.batch_hidden = self.lstm_layer(x, self.batch_hidden)
        else:
            # x = x.view(self.batch * self.traj_len, 1, x.shape[-1], x.shape[-1])
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            x = x.view(x.size(0),1, -1)
            # x = x.view(self.batch, self.traj_len, -1)
            output, self.hidden = self.lstm_layer(x, self.hidden)


        return self.lin2(F.relu(output))

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, 1, self.hidden_dim).to(self.device),
         torch.zeros(self.lstm_layers, 1, self.hidden_dim).to(self.device))

    def init_batch_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.lstm_layers, self.batch, self.hidden_dim).to(self.device),
         torch.zeros(self.lstm_layers, self.batch, self.hidden_dim).to(self.device))

    def clone_hidden(self):
        return (self.hidden[0].clone(),self.hidden[1].clone())

class ReplayBuffer:
    def __init__(self, capacity,  batch = 1, full_episodes_capacity = 1000):
        self.capacity = capacity
        self.batch = batch
        self.full_episodes_capacity = full_episodes_capacity
        self.memory = []
        self.full_episodes_memory = []
        self.position = 0
        self.full_episodes_position = 0

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def add_episode(self, episode):
        if len(self.full_episodes_memory) < self.full_episodes_capacity:
            self.full_episodes_memory.append(None)
        self.full_episodes_memory[self.full_episodes_position]  = episode
        self.full_episodes_position = (self.full_episodes_position + 1) % self.full_episodes_capacity

    def sample(self, batch_size):
        mem_size = len(self.memory)
        batch = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*batch)
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def sample_episode(self):
        episodes = random.sample(self.full_episodes_memory, self.batch)
        # episode = random.sample(self.full_episodes_memory, self.batch)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_is_pad = [],[],[],[],[],[]
        for episode in episodes:
            episode_state, episode_action, episode_reward, episode_next_state, episode_done, episode_is_pad = zip(*episode)
            batch_state.append(episode_state)
            batch_action.append(episode_action)
            batch_reward.append(episode_reward)
            batch_next_state.append(episode_next_state)
            batch_done.append(episode_done)
            batch_is_pad.append(episode_is_pad)
        return batch_state, batch_action, batch_reward, batch_next_state, batch_done, batch_is_pad

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

