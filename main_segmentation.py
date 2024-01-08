# IMPORTS
from preprocessing import *
from classifier import *
from display_results import *
import matplotlib.pyplot as plt

# SET DIRECTORIES
DATA_DIR = 'S:/kaja/ZMO/'
TRAIN_DIR = DATA_DIR+'git_repo/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
RESULTS_TRAIN_DIR = DATA_DIR+'Results/Results_Training/'

# GLOBAL VARIABLES
SLICE = 95  # slice to display superpixels on
NUM_SEGMENTS = 500  # number of segments obtained from SLIC
NUM_IMAGES = 369
NUM_FEATURES = 12
VERSION = 'b'

# FOR ALL IMAGES, COMPUTE SUPERPIXELS AND DESCRIPTORS

def preprocess(in_dir, out_dir):
    count = 0
    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for dir in dirnames:

            ### LOAD DATA ###
            img, mask = load_data(dirpath, dir)

            ### SLIC SUPERPIXELS ###
            #superpixels, overlay = slic_(img[:, :, :], NUM_SEGMENTS, SLICE)
            superpixels = np.load(RESULTS_TRAIN_DIR + dir + '_superpixels_a.npy', allow_pickle=True)  #VERSION

            ### DESCRIPTORS ###
            intensities = intensities_in_superpixels(img, NUM_SEGMENTS, superpixels)
            means, stds, histograms, edges = descriptors(intensities)
            tumor_pixels = count_tumor(superpixels, mask, NUM_SEGMENTS)
            labels = tumor_labeling(tumor_pixels, NUM_SEGMENTS)
            count += 1
            if count == 1:
                np.save(os.path.join(out_dir, dir+'_edges_'+VERSION), edges)

            ### SAVE RESULTS ###
            print(dir)
            #np.save(os.path.join(out_dir, dir+'_superpixels_'+VERSION), superpixels)
            #np.save(os.path.join(out_dir, dir+'_intensities_'+VERSION), np.array(intensities))
            np.save(os.path.join(out_dir, dir+'_means_'+VERSION), means)
            np.save(os.path.join(out_dir, dir+'_stds_'+VERSION), stds)
            np.save(os.path.join(out_dir, dir+'_histograms_'+VERSION), histograms)
            np.save(os.path.join(out_dir, dir+'_labels_'+VERSION), labels)



if __name__ == '__main__':
    #preprocess(TRAIN_DIR, RESULTS_TRAIN_DIR)

    X, y, weights = get_features(RESULTS_TRAIN_DIR, NUM_SEGMENTS, NUM_IMAGES, NUM_FEATURES)
    found_tumors, mislab_nontumors, model = cross_val(X, y, weights)

    display_results(model, RESULTS_TRAIN_DIR, TRAIN_DIR, NUM_FEATURES, NUM_SEGMENTS, found_tumors, mislab_nontumors)

