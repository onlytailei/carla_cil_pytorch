import itertools

import h5py
import matplotlib.pyplot as plt
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


# vpcom Special function for visualization
def plotSpecialTool(data, labels, samples2Visualize=12, factors=[2, 6],
                    grayFlag=False, figSize=(12, 3), fontsize=7):
    # samples2Visualize = 12 # sample 12 random number
    # factors = [2,6] # indicate two factors for number of samples
    assert np.prod(np.array(factors)) == samples2Visualize, \
            "%rx%r is not equal to %r" % (factors[0],
                                          factors[1],
                                          samples2Visualize)
    figure = plt.figure(figsize=figSize)
    nLimit = data.shape[0]
    for i in range(1, samples2Visualize+1):
        img = figure.add_subplot(factors[0], factors[1], i)
        # randomly sample an image from train set
        imgID = np.random.randint(nLimit-1)
        image = data[imgID]
        if grayFlag:
            plt.imshow(image.reshape(image.shape[0],
                                     image.shape[1]),
                       cmap=plt.get_cmap('gray'))
        else:
            plt.imshow(image)
        img.set_title(["{:06.4f}".format(x) for x in labels[imgID]],
                      fontsize=fontsize)
        plt.axis('off')


def genData(fileNames, batchSize=200):
    batchX = np.zeros((batchSize, 88, 200, 3))
    batchY = np.zeros((batchSize, 28))
    idx = 0
    while True:  # to make sure we never reach the end
        counter = 0
        while counter <= batchSize-1:
            idx = np.random.randint(len(fileNames)-1)
            try:
                data = h5py.File(fileNames[idx], 'r')
            except:
                print(idx, fileNames[idx])

            dataIdx = np.random.randint(200-1)
            batchX[counter] = data['rgb'][dataIdx]
            batchY[counter] = data['targets'][dataIdx]
            counter += 1
            data.close()
        yield (batchX, batchY)


def genBranch(fileNames, branchNum=3, batchSize=200):
    batchX = np.zeros((batchSize, 88, 200, 3))
    batchY = np.zeros((batchSize, 28))
    idx = 0
    while True:  # to make sure we never reach the end
        counter = 0
        while counter <= batchSize-1:
            idx = np.random.randint(len(fileNames)-1)
            try:
                data = h5py.File(fileNames[idx], 'r')
            except:
                print(idx, fileNames[idx])

            dataIdx = np.random.randint(200-1)
            if data['targets'][dataIdx][24] == branchNum:
                batchX[counter] = data['rgb'][dataIdx]
                batchY[counter] = data['targets'][dataIdx]
                counter += 1
                data.close()
        yield (batchX, batchY)

st = lambda aug: iaa.Sometimes(0.4, aug)
oc = lambda aug: iaa.Sometimes(0.3, aug)
rl = lambda aug: iaa.Sometimes(0.09, aug)
seq = iaa.Sequential([
        rl(iaa.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 1.5
        rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)), # add gaussian noise to images
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iaa.Add((-40, 40), per_channel=0.5)), # change brightness of images (by -X to Y of original value)
        st(iaa.Multiply((0.10, 2.5), per_channel=0.2)), # change brightness of images (X-Y% of original value)
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast
        #rl(iaa.Grayscale((0.0, 1))), # put grayscale
], random_order=True)
