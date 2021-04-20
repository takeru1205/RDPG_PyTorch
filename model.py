import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, history, hidden):
        x, hidden = self.lstm(history)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x, hidden
