from time import gmtime, strftime
import os
import numpy as np
import keras
from keras.layers import Conv2D, UpSampling2D, \
    Flatten, Activation, Reshape, MaxPooling2D, Input, merge, Dense
from keras.models import Model
import pandas as pd
import keras.metrics
import tensorflow as tf
import utils as myutils
from synthdataio import data_generator_landmarks as sdatapts
from realdataio import data_generator_landmarks as rdatapts
import tqdm
import pickle

################################################
def buildModelLandmarks(inpShape=(256, 256, 3), numPts=8, ptsDim=2):
    ii = Input(shape=inpShape)
    x = Conv2D(32, (3, 3), activation='relu')(ii)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(numPts*ptsDim, activation=None)(x)
    model = Model(inputs=[ii], outputs=[x])
    return model
################################################


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)

    imgShp=(256,256,3)

    print ('Build new model...')
    model = buildModelLandmarks(inpShape=imgShp, numPts=8, ptsDim=2)
    with tf.name_scope('Keras_Optimizer'):
        adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='mean_squared_error')

    model.summary()

    realdatacsv = pd.read_csv("real_data_landmarks.csv")
    rdatatrain = realdatacsv[realdatacsv['subsetid'] == 0]
    rdataval = realdatacsv[realdatacsv['subsetid'] == 1]
    rdatatest = realdatacsv[realdatacsv['subsetid'] == 2]
    synthdatacsv = pd.read_csv('synth_data.csv')
    sdatatrain = synthdatacsv[synthdatacsv['subsetid'] == 0]
    sdataval = synthdatacsv[synthdatacsv['subsetid'] == 1]
    sdatatest = synthdatacsv[synthdatacsv['subsetid'] == 2]

    # 8 points are selected: eye corners, and 4 around mouth
    selection_real = np.array([0, 1, 2, 3, 7, 8, 9, 10])
    selection_synth = np.array([0, 3, 6, 9, 58, 55, 61, 64])

    genRealTrain = rdatapts(rdatatrain, selection_real, batchSize=8, pimgSize=256, augment=True)
    genSynthTrain = sdatapts(sdatatrain, selection_synth, batchSize=8, pimgSize=256, augment=True)

    genRealVal = rdatapts(rdataval, selection_real, batchSize=8, pimgSize=256, augment=False)
    genSynthVal = sdatapts(sdataval, selection_synth, batchSize=8, pimgSize=256, augment=False)

    genRealTest = rdatapts(rdatatest, selection_real, batchSize=8, pimgSize=256, augment=False)
    genSynthTest = sdatapts(sdatatest, selection_synth, batchSize=8, pimgSize=256, augment=False)

    # dictionary to save losses
    valScores = {'realval':[], 'synthval':[], 'realtrain':[], 'synthtrain':[]}

    # folder for live demo plotting
    monitordir = 'outputs/images_monitor_landmarks'
    if not os.path.exists(monitordir):
        os.makedirs(monitordir)

    ep = 0
    numEpochs = 300
    numIterPerEpochTrn = 500
    numIterPerEpochVal = 300
    numIterValReal = 300

    for ep in range(ep, ep+numEpochs):

        # epoch of training, select another generator or concatenate batches to mix data domains
        for it in tqdm.tqdm(range(numIterPerEpochTrn)):
            x1, y1 = next(genRealTrain)
            # x2, y2 = next(genSynthTrain)
            # x1 = np.concatenate([x1,x2])
            # y1 = np.concatenate([y1,y2])
            model.train_on_batch(x1, y1)

        # validate and save scores
        scores = model.evaluate_generator(genRealTrain, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['realtrain'].append(scores)
        scores = model.evaluate_generator(genRealVal,numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['realval'].append(scores)
        scores = model.evaluate_generator(genSynthTrain, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['synthtrain'].append(scores)
        scores = model.evaluate_generator(genSynthVal, numIterPerEpochVal, use_multiprocessing=True, workers=4)
        valScores['synthval'].append(scores)

        # saving image demos
        myutils.demo_pred_landmarks(model=model, datagen=genRealVal, outfolder=monitordir,
                                    fname='plot_real_{}.png'.format(ep), num=4)
        myutils.demo_pred_landmarks(model=model, datagen=genSynthVal, outfolder=monitordir,
                                    fname='plot_synth_{}.png'.format(ep), num=4)

        # saving model after each epoch
        modelsdir = 'outputs/models_landmarks'
        if not os.path.exists(modelsdir):
            os.makedirs(modelsdir)
        model.save(os.path.join(modelsdir, 'mdl_ep{}_{}.h5'.format(ep, strftime("%Y-%m-%d_%H-%M-%S", gmtime()))))
        # rewrite scores after each epoch + backup so interrupting writing won't ruin all records
        pickle.dump(valScores, open(os.path.join(modelsdir, 'valScores.pkl'), 'wb'))
        pickle.dump(valScores, open(os.path.join(modelsdir, 'valScores_bkup.pkl'), 'wb'))


