from time import gmtime, strftime
import os
import matplotlib.pyplot as plt
import skimage.io as skio
import numpy as np
import keras
from keras.layers import Conv2D, UpSampling2D, \
    Flatten, Activation, Reshape, MaxPooling2D, Input, merge
from keras.models import Model
import keras.optimizers as kopt
import keras.losses
import pandas as pd
import keras.metrics
from keras.utils.vis_utils import plot_model as kplot
import tensorflow as tf
import utils as myutils
from synthdataio import data_generator_segmentation as sdata
from synthdataio import data_generator_bgfg as sdata2
from realdataio import data_generator_simple as rdata
import tqdm
import pickle

################################################
# tip my hat off to Alexander Kalinovsky for U-Net :^)
def buildModelUNet(inpShape=(256, 256, 3), numCls=3, numConv=2, kernelSize=3, numFlt=8, ppad='same', numSubsampling=5, isDebug=False):
    dataInput = Input(shape=inpShape)
    fsiz = (kernelSize, kernelSize)
    psiz = (2, 2)
    x = dataInput
    # -------- Encoder --------
    lstMaxPools = []
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2**cc), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        lstMaxPools.append(x)
        x = MaxPooling2D(pool_size=psiz)(x)
    # -------- Decoder --------
    for cc in range(numSubsampling):
        for ii in range(numConv):
            x = Conv2D(filters=numFlt * (2 ** (numSubsampling - 1 -cc)), kernel_size=fsiz, padding=ppad, activation='relu')(x)
        x = UpSampling2D(size=psiz)(x)
        if cc< (numSubsampling-1):
            x = keras.layers.concatenate([x, lstMaxPools[-1 - cc]], axis=-1)
    x = Conv2D(filters=numCls, kernel_size=(1, 1), padding='valid')(x)
    x = Reshape([-1, numCls])(x)
    x = Activation('softmax')(x)
    retModel = Model(dataInput, x)
    if isDebug:
        retModel.summary()
        fimg_model = 'model_FCN_UNet.png'
        kplot(retModel, fimg_model, show_shapes=True)
        plt.imshow(skio.imread(fimg_model))
        plt.show()
    return retModel


################################################
if __name__ == '__main__':
    # paths to csvs
    csvSynth = 'synth_data.csv'
    csvSynth = pd.read_csv(csvSynth)
    csvSynthTrain = csvSynth[csvSynth['subsetid'] == 0]
    csvSynthVal = csvSynth[csvSynth['subsetid'] == 1]
    csvReal = 'real_data_segmentation.csv'
    csvReal = pd.read_csv(csvReal)
    csvRealTrain = csvReal[csvReal['subsetid'] == 0]
    csvRealVal = csvReal[csvReal['subsetid'] == 1]
    numCls  = 2 # nCls for U-Net constructor
    imgSiz  = 256
    imgShp  = (imgSiz, imgSiz, 3)
    batchSize = 16
    numEpochs = 160
    numSamplesTrn = len(csvSynthTrain)
    numSamplesVal = len(csvSynthVal)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)

    print ('Build new model...')
    model = buildModelUNet(inpShape=imgShp, numCls=numCls, isDebug=False)
    with tf.name_scope('Keras_Optimizer'):
        modelOptimizer = kopt.Adam(lr=0.0001)
        model.compile(optimizer=modelOptimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    numIterPerEpochTrn = 3
    numIterPerEpochVal = 3
    numIterValReal = 3

    # generators for synth data
    # 5 classes are skin, eyes, teeth, tongue, bg
    genSynthTrain = sdata(csvSynthTrain, numCls=5, pimgSize=imgSiz, batchSize=batchSize, augment=True)
    genSynthVal = sdata(csvSynthVal, numCls=5, pimgSize=imgSiz, batchSize=batchSize, augment=False)
    genSynthBgFgTrain = sdata2(csvSynthTrain, pimgSize=imgSiz, batchSize=batchSize, augment=True)
    genSynthBgFgVal = sdata2(csvSynthTrain, pimgSize=imgSiz, batchSize=batchSize, augment=False)

    # real data has only 2 classes
    genRealTrain = rdata(csvRealTrain,
                         numCls=2,
                         batchSize=batchSize,
                         pimgSize=256,
                         augment=True)

    genRealVal = rdata(csvRealVal,
                       numCls=2,
                       batchSize=batchSize,
                       pimgSize=256,
                       augment=False)

    # savinglosses
    valScores = {'realval':[], 'synthval':[], 'realtrain':[], 'synthtrain':[]}

    monitordir = 'outputs/images_monitor_segmentation'
    if not os.path.exists(monitordir):
        os.makedirs(monitordir)

    ep = 0
    #REAL+SYNTH TRAINING
    for ep in range(ep, ep+numEpochs):
        # epoch of training
        for it in tqdm.tqdm(range(numIterPerEpochTrn)):
            x1, y1 = next(genRealTrain)
            x2, y2 = next(genSynthBgFgTrain)
            x = np.concatenate([x1,x2])
            y = np.concatenate([y1,y2])
            model.train_on_batch(x, y)

        # validate
        scores = model.evaluate_generator(genRealTrain, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['realtrain'].append(scores)
        scores = model.evaluate_generator(genRealVal, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['realval'].append(scores)
        scores = model.evaluate_generator(genSynthBgFgTrain, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['synthtrain'].append(scores)
        scores = model.evaluate_generator(genSynthBgFgVal, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['synthval'].append(scores)

        myutils.demo_prediction(model, csvRealVal, outfolder=monitordir,
                                fname='plot_real_{}.png'.format(ep), imSize=256, nCls=2)
        myutils.demo_prediction(model, csvSynthVal, outfolder=monitordir,
                                fname='plot_synth_{}.png'.format(ep), imSize=256, nCls=2)

        # saving model every epoch
        modelsdir = 'outputs/models_segmentation'
        if not os.path.exists(modelsdir):
            os.makedirs(modelsdir)
        model.save(os.path.join(modelsdir, 'mdl_ep{}_{}.h5'.format(ep, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))))
        # rewrite scores after each epoch + backup so interrupting writing won't ruin all records
        pickle.dump(valScores, open(os.path.join(modelsdir, 'valScores.pkl'), 'wb'))
        pickle.dump(valScores, open(os.path.join(modelsdir, 'valScores_bkup.pkl'), 'wb'))



