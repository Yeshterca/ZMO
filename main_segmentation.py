# IMPORTS
from preprocessing import *
#from classifier import *
import matplotlib.pyplot as plt

# SET DIRECTORIES
DATA_DIR = 'C:/Users/kajin/Documents/_/3/ZMO/sm/archiven/'
TRAIN_DIR = DATA_DIR+'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALID_DIR = DATA_DIR+'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'
RESULTS_TRAIN_DIR = DATA_DIR+'Results/Results_Training/'
RESULTS_VALID_DIR = DATA_DIR+'Results/Results_Validation/'

# GLOBAL VARIABLES
SLICE = 95  # slice to display superpixels on
NUM_SEGMENTS = 500  # number of segments obtained from SLIC
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


preprocess(TRAIN_DIR, RESULTS_TRAIN_DIR)

#preprocess(VALID_DIR, RESULTS_VALID_DIR)

# CLASSIFIER

#features = get_features(RESULTS_TRAIN_DIR, NUM_SEGMENTS)

    #inputs: means, stds, tumor_labeling
    #output: tumor_labeling


