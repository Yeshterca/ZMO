import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset

NUM_FEATURES = 12

class Data(Dataset):
    def __init__(self, X, y, weights):
        assert X.shape[0] == y.shape[0] == len(weights)
        self.len = X.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.weights = torch.from_numpy(weights.astype(np.float32))

    def __getitem__(self, index):
        return self.X[index, :], self.y[index], self.weights[index]

    def __len__(self):
        return self.len


# CLASSIFICATION
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = 30

        self.lin1 = nn.Linear(NUM_FEATURES, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.lin3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.lin3(x)
        x = torch.nn.functional.sigmoid(x)
        return x

