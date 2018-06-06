import skimage.io as skio
import skimage.transform as sktf
import skimage.color as skcol
import numpy as np
from keras.utils import np_utils
from augmenter import *
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, threshold_mean
import pandas as pd
import utils


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


def read_keypoints_real(pkp, selection):
    pts = pd.read_csv(pkp, header=None).as_matrix()[selection, :2]
    return pts


def read_msk(pmsk, pimsiz = None):
    if isinstance(pmsk, str) or isinstance(pmsk, unicode):
        pmsk = skio.imread(pmsk)
    if pimsiz is not None and pimsiz != pmsk.shape[0] and pimsiz != pmsk.shape[1]:
        pmsk = sktf.resize(pmsk, (pimsiz, pimsiz), order=0, preserve_range=True).astype(np.uint8)
    return pmsk[:,:,0]


def data_generator_landmarks(dataCSV, selection, batchSize=4, pimgSize=256, augment=True):
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
            timg = read_img(pimg, pimgSize).astype(np.uint8)[:, :, :3]
            tkp = read_keypoints_real(pkp, selection)
            if np.random.rand() < 0.5: # we always augment flips
                timg, tkp = utils.fliplr(timg, tkp)
            tkp = (tkp*256).astype(np.int32)
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


def data_generator_simple(dataCSV, numCls, batchSize=4, pimgSize=256, augment=True):
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
            timg = read_img(pimg, pimgSize).astype(np.uint8)[:,:,:3]
            tmsk = read_msk(pmsk, pimgSize).astype(np.uint8)
            th = 70
            tmsk[tmsk<th] = 0
            tmsk[tmsk>=th] = 1
            assert len(tmsk.shape) == 2, 'error: mask has many channels'
            if dataX is None:
                dataX = np.zeros([batchSize] + list(timg.shape), dtype=np.uint8)
                dataY = np.zeros([batchSize] + list(tmsk.shape), dtype=np.uint8)
            dataX[ii] = timg
            dataY[ii] = tmsk
        # augment and flatten dataY
        if augment:
            seq_geom = aug.seq_geom.to_deterministic()
            seq_color = aug.seq_color.to_deterministic()
            dataX, dataY = list(seq_geom.augment_batches(batches=[dataX, dataY], background=False))
            dataX, = list(seq_color.augment_batches(batches=[dataX], background=False))
            dataY = np_utils.to_categorical(dataY, numCls).reshape(batchSize, pimgSize*pimgSize, numCls)
            dataX = dataX.astype(np.float32)/127.5 - 1.0
        else:
            dataY = np_utils.to_categorical(dataY, numCls).reshape(batchSize, pimgSize * pimgSize, numCls)
            dataX = dataX.astype(np.float32) / 127.5 - 1.0
        yield dataX, dataY
