import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skcol
import numpy as np
from keras.utils import np_utils
from augmenter import *
import matplotlib.pyplot as plt
import utils
import pandas as pd


def read_img(pimg, pimsiz = None):
    if isinstance(pimg, str) or isinstance(pimg, unicode):
        pimg = skio.imread(pimg)
    if pimsiz is not None and pimsiz != pimg.shape[0] and pimsiz != pimg.shape[1]:
        pimg = sktf.resize(pimg, (pimsiz, pimsiz), order=1, preserve_range=True)
    else:
        pimg = pimg.astype(np.float32)
    if pimg.ndim < 3:
        pimg = skcol.gray2rgb(pimg)
    #pimg = pimg/127.5 - 1.0
    return pimg


def labels_from_colored(img):
    """dataset specific procedure; also known as 'crutch'"""
    true_colors = [[0, 0, 0, 0],
                   [1, 0, 0, 1],
                   [0.91, 0.91, 0, 1],
                   [0.74, 0.54, 0, 1],
                   [0.54, 0.54, 0.88, 1],
                   [0, 0, 1, 1],
                   [0.91, 0.91, 0.91, 1]]
    classes_transform = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}
    # nameslabels = ['bg', 'skin', 'hair', 'eyebrow', 'tongue', 'eye', 'teeth']
    labeled = np.zeros(img.shape[:-1])
    for i in range(len(true_colors)):
        eq = np.all(np.isclose(img.astype(np.float32) / 255, true_colors[i][:3], atol=0.01), axis=-1)
        labeled[eq] = classes_transform[i]
    return labeled


def labels_bg_fg(img):
    true_colors = [[0, 0, 0, 0],
                   [1, 0, 0, 1],
                   [0.91, 0.91, 0, 1],
                   [0.74, 0.54, 0, 1],
                   [0.54, 0.54, 0.88, 1],
                   [0, 0, 1, 1],
                   [0.91, 0.91, 0.91, 1]]
    classes_transform = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1} # bg and fg only
    # nameslabels = ['bg', 'skin', 'hair', 'eyebrow', 'tongue', 'eye', 'teeth']
    labeled = np.zeros(img.shape[:-1])
    for i in range(len(true_colors)):
        eq = np.all(np.isclose(img.astype(np.float32) / 255, true_colors[i][:3], atol=0.01), axis=-1)
        labeled[eq] = classes_transform[i]
    return labeled


def read_msk(pmsk, pimsiz = None):
    if isinstance(pmsk, str) or isinstance(pmsk, unicode):
        pmsk = skio.imread(pmsk)
    if pimsiz is not None and pimsiz != pmsk.shape[0] and pimsiz != pmsk.shape[1]:
        pmsk = sktf.resize(pmsk, (pimsiz, pimsiz), order=0, preserve_range=True).astype(np.uint8)
    return pmsk


def read_keypoints_synth(path, selection):
    pts = pd.read_csv(path, header=None).as_matrix()[selection, :2]
    pts = utils.fixlandmarks(pts)
    return pts


def data_generator_landmarks(dataCSV, selection, batchSize, pimgSize=256, augment=True):
    aug = Augmenter()
    numSamples = len(dataCSV)
    pathImgs = np.array(dataCSV['img'])
    pathKps = np.array(dataCSV['keypoints'])
    idxRange = np.array(range(numSamples))
    while True:
        rndIdx = np.random.permutation(idxRange)[:batchSize]
        dataX = None
        dataY = None
        for ii, iidx in enumerate(rndIdx):
            pimg = pathImgs[iidx]
            pkp = pathKps[iidx]
            timg = read_img(pimg, pimgSize).astype(np.uint8)
            tkp = read_keypoints_synth(pkp, selection)*256
            timg, tkp = utils.fixbb(timg, tkp)
            timg = utils.background_blend(timg)
            tkp = tkp/256
            if np.random.rand() < 0.5: # we always augment flips
                timg, tkp = utils.fliplr(timg, tkp)
            tkp = (tkp*256).astype(np.int32)
            timg = utils.to_uint8(timg)
            if dataX is None:
                dataX = np.zeros([batchSize] + list(timg.shape), dtype=np.uint8)
                dataY = np.zeros([batchSize] + list(tkp.shape), dtype=np.int32)
            dataX[ii] = timg
            dataY[ii] = tkp
        # augment and flatten dataY
        if augment:
            keypoints = [ia.KeypointsOnImage.from_coords_array(ptsi, (256, 256)) for ptsi in dataY]
            seq_geom = aug.seq_geom.to_deterministic()
            seq_color = aug.seq_color.to_deterministic()
            dataX, keypoints = list(seq_geom.augment_batches(batches=[dataX, keypoints], background=False))
            dataX, = list(seq_color.augment_batches(batches=[dataX], background=False))
            dataX = dataX.astype(np.float32) / 127.5 - 1.0
            dataY = np.array([kp.get_coords_array() for kp in keypoints])
        else:
            dataX = dataX.astype(np.float32) / 127.5 - 1.0
        yield dataX, dataY.reshape((batchSize,-1)).astype(np.float32)/256


def data_generator_bgfg(dataCSV, batchSize=4, pimgSize=256, augment=False):
    numCls=2
    aug = Augmenter()
    numSamples = len(dataCSV)
    pathImgs = np.array(dataCSV['img'])
    pathMsks = np.array(dataCSV['mask'])
    idxRange = np.array(range(numSamples))
    while True:
        rndIdx = np.random.permutation(idxRange)[:batchSize]
        dataX = None
        dataY = None
        for ii, iidx in enumerate(rndIdx):
            pimg = pathImgs[iidx]
            pmsk = pathMsks[iidx]
            timg = read_img(pimg, pimgSize)
            timg = utils.background_blend(timg)
            timg = utils.to_uint8(timg)
            tmsk = read_msk(pmsk, pimgSize)[...,:3]
            if dataX is None:
                dataX = np.zeros([batchSize] + list(timg.shape), dtype=np.uint8)
                dataY = np.zeros([batchSize] + list(tmsk.shape), dtype=np.uint8)
            dataX[ii] = timg
            dataY[ii] = tmsk
        # augment and flatten dataY
        if augment:
            seq_geom = aug.seq_geom.to_deterministic()
            seq_color = aug.seq_color.to_deterministic()
            #seq_geom.augment_batches()
            dataX, dataY = list(seq_geom.augment_batches(batches=[dataX, dataY], background=False))
            dataX, = list(seq_color.augment_batches(batches=[dataX], background=False))
            dataY = labels_bg_fg(dataY)
            dataY = np_utils.to_categorical(dataY, numCls).reshape(batchSize, pimgSize*pimgSize, numCls)
            dataX = dataX.astype(np.float32)/127.5 - 1.0
        else:
            dataY = labels_bg_fg(dataY)
            dataY = np_utils.to_categorical(dataY, numCls).reshape(batchSize, pimgSize * pimgSize, numCls)
            dataX = dataX.astype(np.float32) / 127.5 - 1.0
        yield dataX, dataY


def data_generator_segmentation(dataCSV, numCls, batchSize=4, pimgSize=256, augment=False):
    aug = Augmenter()
    numSamples = len(dataCSV)
    pathImgs = np.array(dataCSV['img'])
    pathMsks = np.array(dataCSV['mask'])
    idxRange = np.array(range(numSamples))
    while True:
        rndIdx = np.random.permutation(idxRange)[:batchSize]
        dataX = None
        dataY = None
        for ii, iidx in enumerate(rndIdx):
            pimg = pathImgs[iidx]
            pmsk = pathMsks[iidx]
            timg = read_img(pimg, pimgSize)
            timg = utils.background_blend(timg)
            timg = utils.to_uint8(timg)
            tmsk = read_msk(pmsk, pimgSize)[...,:3]
            if dataX is None:
                dataX = np.zeros([batchSize] + list(timg.shape), dtype=np.uint8)
                dataY = np.zeros([batchSize] + list(tmsk.shape), dtype=np.uint8)
            dataX[ii] = timg
            dataY[ii] = tmsk
        # augment and flatten dataY
        if augment:
            seq_geom = aug.seq_geom.to_deterministic()
            seq_color = aug.seq_color.to_deterministic()
            #seq_geom.augment_batches()
            dataX, dataY = list(seq_geom.augment_batches(batches=[dataX, dataY], background=False))
            dataX, = list(seq_color.augment_batches(batches=[dataX], background=False))
            dataY = labels_from_colored(dataY)
            dataY = np_utils.to_categorical(dataY, numCls).reshape(batchSize, pimgSize*pimgSize, numCls)
            dataX = dataX.astype(np.float32)/127.5 - 1.0
        else:
            dataY = labels_from_colored(dataY)
            dataY = np_utils.to_categorical(dataY, numCls).reshape(batchSize, pimgSize * pimgSize, numCls)
            dataX = dataX.astype(np.float32) / 127.5 - 1.0
        yield dataX, dataY