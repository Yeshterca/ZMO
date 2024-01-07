# IMPORTS
import numpy as np
import nibabel as nib
import os
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries


# LOAD DATA
def load_data(dirpath, dir):
    """
    in the path directory, go through all folders, in each find t2 and seg image

    :return: images, masks = x, y, slices, number of image """

    mask = np.zeros([240, 240, 155]) #369
    img = np.zeros([240, 240, 155])
    for file in os.listdir(dirpath + dir):
        if file.endswith('t2.nii'):
            img = nib.load(dirpath + dir + '/' + file).get_fdata()
        elif file.endswith('seg.nii'):
            mask = nib.load(dirpath + dir + '/' + file).get_fdata()
    return img, mask


# SLIC
def slic_(image, num_segments, slice_n):
    """
    computes superpixels using SLIC

    :param image ... 3D image (x,y,slices)
    :param num_segments ... number of labels in the output image
    :param slice_n ... number of slice on which to display the overlay

        - channel_axis ... None since we work with BW images, although it is recommended to work with channels
        - compactness ... while none channels, recommended lower
        - enforce_connectivity ... whether the general segments are connected ... idk, try :)

    :return: segments (240,240,155)
    :return: marked (240,240) ... image with an overlay of computed superpixels ! has to be computed on 2D image
                                    (i pick one slice)
    """

    segments = slic(image, num_segments, compactness=0.2, channel_axis=None, enforce_connectivity = True)
    marked = mark_boundaries(image[:, :, slice_n], segments[:, :, slice_n])
    return segments, marked


# DESCRIPTORS
def intensities_in_superpixels(image, num_segments, segments):
    """
    goes through all pixels of the image, intensity of each pixel is saved under corresponding superpixel label

    :param image ... 3D image (x, y, slices):
    :param num_segments ... number of superpixels
    :param segments ... superpixels (labels), (x, y, slices)
    :return: intensities ... list of arrays, each array contains intensity values of each pixel that correspond to
                                one superpixel"""

    # intensities of pixel that correspond to the same segment are put together
    intensities = [[] for _ in range(num_segments)]
    for (x, y, z), label in np.ndenumerate(segments):
        intensities[label].append(image[x, y, z])
    intensities = [np.array(ints) for ints in intensities]

    return intensities


def descriptors(intensities, nbins=10, range=(0, 1200)):
    """
    calculates mean, std and histogram of intensities in every superpixel

    :param intensities ... list of arrays, each array contains intensity values of each pixel that corresponds to
                            one superpixel
    :param nbins ... number of bins of histogram
    :param range ... range of the histogram

    :return: means, stds, histogram
    """

    # if empty array, None ! do not skip (indexing)
    means = [np.mean(ints) if ints.size > 0 else None for ints in intensities]
    stds = [np.std(ints) if ints.size > 0 else None for ints in intensities]

    histograms = np.zeros((nbins, len(intensities)))
    edges = 0
    i = 0
    for ints in intensities:
        if ints.size > 0:
            histograms[:, i], edges = np.histogram(np.array(ints), bins=nbins, range=range)
        else:
            histograms[:, i] = None
        i += 1

    return means, stds, histograms, edges


def count_tumor(superpixels, mask, num_segments):
    """
    count all pixels in each superpixel that are labeled as tumor / not-tumor in mask

    :param superpixels: superpixels (labels), (x, y, slices)
    :param mask: true labeling, (x, y, slices)
    :param num_segments: number of superpixels
    :return: tumor_pixels ... np.array(num_segments,2), sum of all pixels in each superpixel that are labeled as
                                tumor / not-tumor in mask
    """
    tumor_pixels = np.zeros((num_segments, 2))
    for (x, y, z), label in np.ndenumerate(superpixels):
        if mask[x, y, z] > 0.:  # TODO only some labels?
            tumor_pixels[label, 0] += 1
        else:
            tumor_pixels[label, 1] += 1

    return tumor_pixels


def tumor_labeling(tumor_pixels, num_segments):
    """
    returns an array of labels corresponding to each superpixel

    :param tumor_pixels: np.array(num_segments,2)
    :param num_segments: number of superpixels
    :return: np.array of 0 / 1 labels
    """
    labels = np.zeros([num_segments, 1])
    for i in range(num_segments):
        if np.sum(tumor_pixels[i, :]) == 0:
            labels[i] = 0.5
            continue
        ratios = tumor_pixels[i, 0] / np.sum(tumor_pixels[i, :])
        if ratios > 0.9:
            labels[i] = 1
        elif ratios < 0.1:
            labels[i] = 0
        else:
            labels[i] = 0.5

    return labels
