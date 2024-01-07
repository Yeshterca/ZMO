# IMPORTS
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
#import pytorch_lightning as pl

# TODO remove
DATA_DIR = 'C:/Users/kajin/Documents/_/3/ZMO/sm/archiven/'
RESULTS_TRAIN_DIR = DATA_DIR+'Results/Results_Training/'
NUM_SEGMENTS = 500
NUM_IMAGES = 369
NUM_FEATURES = 3

# LOAD FEATURES


def get_features(features_dir, num_segments, nbins):
    X = np.zeros([NUM_IMAGES, num_segments, NUM_FEATURES, nbins])
    y = np.zeros([NUM_IMAGES, num_segments])
    num = 0
    i = 0
    for file in os.listdir(features_dir):
        if 'means' in file:
            X[num, :, 0, 0] = np.load((features_dir+file), allow_pickle=True)
            i += 1
        elif 'stds' in file:
            X[num, :, 1, 0] = np.load((features_dir+file), allow_pickle=True)
            i += 1
        elif 'histograms' in file:
            X[num, :, 3, :] = np.load((features_dir+file), allow_pickle=True)
            i += 1
        elif 'labels' in file:
            y[num, :] = np.squeeze(np.load((features_dir+file)))
            i += 1

        if i == 3:
            num += 1
            i = 0

    X = X.reshape((NUM_IMAGES * num_segments, NUM_FEATURES))
    y = y.reshape((NUM_IMAGES * num_segments,))

    good = np.where(y != 0.5)
    X = X[good, :][0, :, :]
    y = y[good]

    return X, y


class Data(Dataset):
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.len = X.shape[0]
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

    def __len__(self):
        return self.len

# # TODO remove
# X, y = get_features(RESULTS_TRAIN_DIR, NUM_SEGMENTS)
# print(X.dtype)
# print(np.shape(X))

# CLASSIFICATION
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        hidden_dim = 10

        self.lin1 = nn.Linear(NUM_FEATURES, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        x = torch.nn.functional.sigmoid(x)
        return x


model = NeuralNetwork()

learning_rate = 0.1
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100  #TODO
batch_size = 1000
loss_values = []

X, y = get_features(RESULTS_TRAIN_DIR, NUM_SEGMENTS)
train_data = Data(X, y)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    print("epoch ", epoch)
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")

"""
Training Complete
"""
