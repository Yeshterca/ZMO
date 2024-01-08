# IMPORTS
import numpy as np
import os
import matplotlib.pyplot as plt
from model import *

# VARIABLES
device = torch.device('cuda:0')
NUM_EPOCHS = 400
LEARNING_RATE = 1e-3
FOLDS = 10
TRAIN_RATIO = 0.9


# LOAD FEATURES
def get_features(features_dir, num_segments, num_images, num_features):
    """
    Load data from features_dir and processes them for training.
    :param features_dir ... directory with precomputed features
    :param num_segments ... number of superpixels
    :param num_images ... number of images
    :param num_features ... number of features (mean, std, histogram bins)
    :return: X ... matrix of all features for all images
    :return: y ... vector of labels
    :return: weights ... vector of adjusted weights for training
    """

    X = np.zeros([num_images, num_segments, num_features])
    y = np.zeros([num_images, num_segments])
    num = 0
    i = 0
    for file in os.listdir(features_dir):
        if 'means_b' in file:
            X[num, :, 0] = np.load((features_dir + file), allow_pickle=True)
            i += 1
        elif 'stds_b' in file:
            X[num, :, 1] = np.load((features_dir + file), allow_pickle=True)
            i += 1
        elif 'histograms_b' in file:
            X[num, :, 2:num_features] = np.transpose(
                np.load((features_dir + file), allow_pickle=True))  # TODO transpose check
            i += 1
        elif 'labels_b' in file:
            y[num, :] = np.squeeze(np.load((features_dir + file)))
            i += 1

        if i == 4:
            num += 1
            i = 0

    X = X.reshape((num_images * num_segments, num_features))
    y = y.reshape((num_images * num_segments,))

    # exclude mixed superpixels
    good = np.where(y != 0.5)[0]
    X = X[good, :]
    y = y[good]

    # exclude background
    foreground = np.where(X[:, 1] > 10.)[0]
    X = X[foreground, :]
    y = y[foreground]

    # TODO remove
    # random shuffle data
    r = np.array(range(X.shape[0]))
    np.random.shuffle(r)
    X = X[r, :]
    y = y[r]

    # select smaller dataset
    size = X.shape[0]
    X = X[:size, :]
    y = y[:size]

    # print
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    # calculate weights
    w_pos = (1. / len(pos)) * (size / 2)
    w_neg = (1. / len(neg)) * (size / 2)
    weights = np.empty(size)
    weights[pos] = w_pos
    weights[neg] = w_neg

    return X, y, weights


def training(X, y, weights):
    """
    Train the classifier
    :param X: matrix of all features of all images
    :param y: vector of labels
    :param weights: vector of adjusted weights
    :return: model ... trained model
    """

    loss_values = []
    model = NeuralNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # data to tensors
    X_tens = torch.from_numpy(X.astype(np.float32))
    y_tens = torch.from_numpy(y.astype(np.float32))
    weights_tens = torch.from_numpy(weights.astype(np.float32))

    # send to GPU
    model.to(device)
    X_gpu, y_gpu, weights_gpu = X_tens.to(device), y_tens.to(device), weights_tens.to(device)

    # training
    for epoch in range(NUM_EPOCHS):
        #print("epoch ", epoch)
        pred = model(X_gpu).squeeze()
        loss = nn.BCELoss(weight=weights_gpu)(pred, y_gpu)
        loss_values.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # send to CPU
    model.to('cpu')
    print("Training Completed")

    # plot loss
    plt.plot(loss_values)
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.show()

    return model

def evaluation(model, X, y):
    """
    Evaluate the model
    :param model: trained model
    :param X: matrix of all features of all images
    :param y: vector of labels
    :return: found_tumors ... percentage of found tumors
    :return: mislabeled_nontumors ... percentage of false positives
    """

    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(X.astype(np.float32))).squeeze()
        outputs = np.array(outputs)

    # count statistics
    predictions = np.where(outputs < 0.5, 0, 1)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    correct_pos = np.sum(predictions[pos] == y[pos])
    correct_neg = np.sum(predictions[neg] == y[neg])

    total = y.shape[0]
    correct = np.sum(predictions == y)
    accuracy = correct / total

    found_tumors = correct_pos / len(pos)
    mislabeled_nontumors = 1 - (correct_neg / len(neg))

    # display statistics
    print(f"true labels: 1/all: {np.sum(y == 1) / y.shape[0]}")
    print(f"positive predictions: {np.sum(predictions != 0)}")
    print(f'Accuracy: {accuracy}')
    print(f'Found tumors (%): {found_tumors}')
    print(f'Misslabeled non-tumors (%): {mislabeled_nontumors}')
    print(f"len(pos): {len(pos)}")
    print(f"len(neg): {len(neg)}")
    print()

    return found_tumors, mislabeled_nontumors


def cross_val(X, y, weights):
    """
    Performs cross validation
    :param X: matrix of all features of all images
    :param y: labels
    :param weights: adjusted weights
    :return: found_tumors ... percentage of found tumors
    :return: mislab_nontumors ... percentage of false positives
    :return: model ... trained model
    """
    found_tumors = np.zeros(FOLDS,)
    mislab_nontumors = np.zeros(FOLDS,)

    for fold in range(FOLDS):
        print("\nCross-validation ", fold+1)

        # Split the data
        indices = np.array(range(X.shape[0]))
        np.random.shuffle(indices)
        cutoff = int(TRAIN_RATIO * len(indices))
        train_idx, val_idx = indices[:cutoff], indices[cutoff:]

        # Training
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        weights_train = weights[train_idx]

        model = training(X_train, y_train, weights_train)

        # Validation
        X_val = X[val_idx, :]
        y_val = y[val_idx]

        found_tumors[fold], mislab_nontumors[fold] = evaluation(model, X_val, y_val)

    return found_tumors, mislab_nontumors, model


