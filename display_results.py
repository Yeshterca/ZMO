import numpy as np
import torch
from preprocessing import *
import matplotlib.pyplot as plt

def display_results(model, results_dir, train_dir, num_features, num_segments, found_tumors, mislab_nontumors):
    """
    Display the results of the segmentation.
    :param model: trained model
    :param results_dir: directory of precomputed features
    :param train_dir: directiory of data
    :param num_features: number of features used in the model
    :param num_segments: number of superpixels
    :param found_tumors: percentages of found tumors from  cross validation
    :param mislab_nontumors: percentages of mislabeled non tumors from cross validation
    """

    avg_found = np.mean(found_tumors)
    avg_mislbl = np.mean(mislab_nontumors)

    print("Average found tumors:", avg_found)
    print("Average mislabeled non-tumors:", avg_mislbl)

    image_count = 9
    count = 0
    X = np.zeros([num_segments, num_features])
    y = np.zeros([num_segments])
    for (dirpath, dirnames, filenames) in os.walk(train_dir):
        for dir in dirnames:

            ### LOAD DATA ###
            img_, mask_ = load_data(dirpath, dir)
            superpixels_ = np.load(results_dir + dir + '_superpixels_a.npy', allow_pickle=True)
            means_ = np.load(results_dir + dir + '_means_b.npy', allow_pickle=True)
            stds_ = np.load(results_dir + dir + '_stds_b.npy', allow_pickle=True)
            histograms_ = np.load(results_dir + dir + '_histograms_b.npy', allow_pickle=True)
            labels_ = np.load(results_dir + dir + '_labels_b.npy', allow_pickle=True)

            X[:, 0] = means_
            X[:, 1] = stds_
            X[:, 2:num_features] = np.transpose(histograms_)
            y = labels_

            count += 1
            if count == image_count:
                break
        if count == image_count:
            break

    with torch.no_grad():
        outputs = model(torch.from_numpy(X.astype(np.float32))).squeeze()
        outputs = np.array(outputs)
        predictions = np.where(outputs < 0.5, 0, 1)

    # display image
    im = np.zeros(np.shape(img_))

    for i in range(len(predictions)):
        if predictions[i] == 1:
            # print(i+1)
            im[superpixels_ == (i + 1)] = 1.
    # im[superpixels_ == 1] = 1.x

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(img_[:, :, 80])
    ax2.imshow(im[:, :, 80])
    ax3.imshow(mask_[:, :, 80])

    ax1.title.set_text('Original image')
    ax2.title.set_text('Segmentation result')
    ax3.title.set_text('Correct segmentation')
    plt.show()


