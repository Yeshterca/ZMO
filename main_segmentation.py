# IMPORTS
from preprocessing import *
from classifier import *
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


preprocess(TRAIN_DIR, RESULTS_TRAIN_DIR)


# CLASSIFIER
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


model = NeuralNetwork()
X, y, weights = get_features(RESULTS_TRAIN_DIR, NUM_SEGMENTS, NUM_IMAGES, NUM_FEATURES)
found_tumors, mislab_nontumors = cross_val(X, y, weights)

avg_found = np.mean(found_tumors)
avg_mislbl = np.mean(mislab_nontumors)

print(avg_found)
print(avg_mislbl)


