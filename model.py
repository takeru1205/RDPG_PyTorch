import torch
import torch.nn as nn
import torch.nn.functional as F


init_w = 3e-3

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=obs_dim+action_dim, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, history, hidden):
        x, hidden = self.lstm(history)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x, hidden

class Crtic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(Crtic, self).__init__()
        self.lstm = nn.LSTM(input_size=obs_dim+action_dim,
                            hidden_size=hidden_size,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, history, hidden):
        x, hidden = self.lstm(history, hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, hidden

