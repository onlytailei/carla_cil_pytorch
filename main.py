# Code and Programming by Tai Lei
# 2017-2018 - www.vpcom.ir
# REF:
'''
http://vladlen.info/papers/carla.pdf
http://vladlen.info/papers/conditional-imitation.pdf
https://github.com/carla-simulator/imitation-learning
https://github.com/mvpcom/carlaILTrainer
'''


import sys
import os
import time
import glob

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import keras

from helper import genData, genBranch, seq
from net import Net

# TODO arguments
timeNumberFrames = 1  # 4 # number of frames in each samples
batchSize = 120  # size of batch
valBatchSize = 120  # size of batch for validation set
NseqVal = 5  # number of sequences to use for validation
# training parameters
epochs = 100
samplesPerEpoch = 500
L2NormConst = 0.001
trainScratch = True
trainScrath = True

# Configurations
num_images = 152800  # 200 * 764
memory_fraction = 0.25
image_cut = [115, 510]
dropoutVec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
prefSize = _image_size = (88, 200, 3)
learningRate = 0.0002  # multiplied by 0.5 every 50000 mini batch
iterNum = 294000
beta1 = 0.7
beta2 = 0.85
# Control signal, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
controlInputs = [2, 5, 3, 4]
cBranchesOutList = ['Follow Lane', 'Go Left', 'Go Right',
                    'Go Straight', 'Speed Prediction Branch']

branchConfig = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                ["Speed"]]
params = [trainScratch, dropoutVec, image_cut, learningRate, beta1, beta2,
          num_images, iterNum, batchSize, valBatchSize, NseqVal,
          epochs, samplesPerEpoch, L2NormConst]

# GPU configuration
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.visible_device_list = '0'
config.gpu_options.per_process_gpu_memory_fraction = memory_fraction

# read an example h5 file
datasetDirTrain = '/home/tai/ws/ijrr_2018/carla_cil_dataset/AgentHuman/chosen_weather_train/clearnnoon_h5'
datasetDirVal = '/home/tai/ws/ijrr_2018/carla_cil_dataset/AgentHuman/chosen_weather_test/clearnoon_h5'

# TODO tackle the training dataset
datasetFilesTrain = glob.glob(datasetDirTrain+'*.h5')
datasetFilesVal = glob.glob(datasetDirVal+'*.h5')
print(len(datasetFilesTrain))
print(len(datasetFilesVal))

config = tf.ConfigProto(allow_soft_placement=True)
tf.reset_default_graph()
sessGraph = tf.Graph()

batchListGenTrain = []
batchListGenVal = []
batchListName = []
for i in range(len(branchConfig)):
    with tf.name_scope("Branch_" + str(i)):
        if branchConfig[i][0] == "Speed":
            miniBatchGen = genData(fileNames=datasetFilesTrain,
                                   batchSize=batchSize)
            batchListGenTrain.append(miniBatchGen)
            miniBatchGen = genData(fileNames=datasetFilesVal,
                                   batchSize=batchSize)
            batchListGenVal.append(miniBatchGen)
        else:
            miniBatchGen = genBranch(fileNames=datasetFilesTrain,
                                     branchNum=controlInputs[i],
                                     batchSize=batchSize)
            batchListGenTrain.append(miniBatchGen)
            miniBatchGen = genBranch(fileNames=datasetFilesVal,
                                     branchNum=controlInputs[i],
                                     batchSize=batchSize)
            batchListGenVal.append(miniBatchGen)

with sessGraph.as_default():
    sess = tf.Session(graph=sessGraph, config=config)
    with sess.as_default():

        # build model
        print('Building Net ...')
        netTensors = Net(branchConfig, params, timeNumberFrames, prefSize)
        print(netTensors['output'])

        print('Initialize Variables in the Graph ...')
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
        # restore trained parameters
        if not(trainScratch):
            saver.restore(sess, "test/model.ckpt")
        # op to write logs to Tensorboard
        logsPath = './logs'
        modelPath = './test/'
        summary_writer = tf.summary.FileWriter(logsPath, graph=sessGraph)
        print('Start Training process ...')

        steps = 0
        for epoch in range(epochs):
            tStartEpoch = time.time()
            print("  Epoch:", epoch)
            for j in range(int(num_images/batchSize)):
                steps += 1

                for i in range(0, len(branchConfig)):

                        xs, ys = next(batchListGenTrain[i])

                        # augment images
                        xs = seq.augment_images(xs)

                        # solverList[i]
                        contSolver = netTensors['output']['optimizers'][i]
                        # lossList[i]
                        contLoss = netTensors['output']['losses'][i]
                        inputData = []

                        # Command Control
                        inputData.append(sess.run(tf.one_hot(ys[:, 24], 4)))
                        # Speed
                        inputData.append(ys[:, 10].reshape([batchSize, 1]))

                        # [ inputs['inputImages','inputData'], targets['targetSpeed', 'targetController'],  'params', dropoutVec', output[optimizers, losses, branchesOutputs] ]
                        feedDict = {netTensors['inputs'][0]: xs,
                                    netTensors['inputs'][1][0]: inputData[0],
                                    netTensors['inputs'][1][1]: inputData[1],
                                    netTensors['dropoutVec']: dropoutVec,
                                    netTensors['targets'][0]:
                                        ys[:, 10].reshape([120, 1]),
                                    netTensors['targets'][1]: ys[:, 0:3]}
                        _, loss_value = sess.run([contSolver, contLoss],
                                                 feed_dict=feedDict)

                        # write logs at every iteration
                        feedDict = {netTensors['inputs'][0]: xs,
                                    netTensors['inputs'][1][0]: inputData[0],
                                    netTensors['inputs'][1][1]: inputData[1],
                                    netTensors['dropoutVec']:
                                        [1] * len(dropoutVec),
                                    netTensors['targets'][0]:
                                        ys[:, 10].reshape([120, 1]),
                                    netTensors['targets'][1]: ys[:, 0:3]}
                        summary = merged_summary_op.eval(feed_dict=feedDict)
                        summary_writer.add_summary(
                            summary,
                            epoch*num_images/batchSize+j)

                        print("  Train::: Epoch: %d, Step: %d, \
                              TotalSteps: %d, Loss: %g"
                              % (epoch, epoch * batchSize + j,
                                 steps, loss_value),
                              cBranchesOutList[i])

                        if steps % 10 == 0:
                            xs, ys = next(batchListGenVal[i])
                            contLoss = netTensors['output']['losses'][i]
                            feedDict = {netTensors['inputs'][0]: xs,
                                        netTensors['inputs'][1][0]:
                                            inputData[0],
                                        netTensors['inputs'][1][1]:
                                            inputData[1],
                                        netTensors['dropoutVec']:
                                            [1] * len(dropoutVec),
                                        netTensors['targets'][0]:
                                            ys[:, 10].reshape([120, 1]),
                                        netTensors['targets'][1]: ys[:, 0:3]}
                            loss_value = contLoss.eval(feed_dict=feedDict)
                            print("  Val::: Epoch: %d, Step: %d, \
                                    TotalSteps: %d, Loss: %g"
                                  % (epoch, epoch * batchSize + j,
                                     steps, loss_value),
                                  cBranchesOutList[i])

                if steps % 50 == 0 and steps != 0:  # batchSize
                    print(j % 50, '  Save Checkpoint ...')
                    if not os.path.exists(modelPath):
                        os.makedirs(modelPath)
                    checkpoint_path = os.path.join(modelPath, "model.ckpt")
                    filename = saver.save(sess, checkpoint_path)
                    print("  Model saved in file: %s" % filename)

                # every 50000 step, multiply learning rate by half
                if steps % 50000 == 0 and steps != 0:
                    print("Half the learning rate ....")
                    solverList = []
                    lossList = []
                    trainVars = tf.trainable_variables()
                    for i in range(0, len(branchConfig)):
                        with tf.name_scope("Branch_" + str(i)):
                            if branchConfig[i][0] == "Speed":
                                # we only use the image as input to speed prediction
                                # if not (j == 0):
                                # [ inputs['inputImages','inputData'], targets['targetSpeed', 'targetController'],  'params', dropoutVec', output[optimizers, losses, branchesOutputs] ]
                                # params = [trainScratch, dropoutVec, image_cut, learningRate, beta1, beta2, num_images, iterNum, batchSize, valBatchSize, NseqVal, epochs, samplesPerEpoch, L2NormConst]
                                params[3] = params[3] * 0.5  # update Learning Rate
                                contLoss = tf.reduce_mean(
                                    tf.square(
                                        tf.subtract(
                                            netTensors['output']['branchesOutputs'][-1],
                                            netTensors['targets'][0]))) #+ tf.add_n([tf.nn.l2_loss(v) for v in trainVars]) * L2NormConst
                                contSolver = tf.train.AdamOptimizer(
                                    learning_rate=params[3],
                                    beta1=params[4],
                                    beta2=params[5]).minimize(contLoss)
                                solverList.append(contSolver)
                                lossList.append(contLoss)
                                # create a summary to monitor cost tensor
                                tf.summary.scalar("Speed_Loss", contLoss)
                            else:
                                #if not (j == 0):
                                params[3] = params[3] * 0.5
                                contLoss = tf.reduce_mean(
                                    tf.square(
                                        tf.subtract(netTensors['output']['branchesOutputs'][i],
                                                    netTensors['targets'][1]))) #+ tf.add_n([tf.nn.l2_loss(v) for v in trainVars]) * L2NormConst
                                contSolver = tf.train.AdamOptimizer(
                                    learning_rate=params[3],
                                    beta1=params[4],
                                    beta2=params[5]).minimize(contLoss)
                                solverList.append(contSolver)
                                lossList.append(contLoss)
                                tf.summary.scalar("Control_Loss_Branch_"+str(i), contLoss)

                    # update new Losses and Optimizers
                    print('Initialize Variables in the Graph ...')
                    # merge all summaries into a single op
                    merged_summary_op = tf.summary.merge_all()
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, "test/model.ckpt")  # restore trained parameters

                if steps % 294000 == 0 and steps != 0:
                    # finish the training
                    break
            if steps % 294000 == 0 and steps != 0:
                # finish the training
                print('Finalize the training and Save Checkpoint ...')
                if not os.path.exists(modelPath):
                    os.makedirs(modelPath)
                checkpoint_path = os.path.join(modelPath, "model.ckpt")
                filename = saver.save(sess, checkpoint_path)
                print("  Model saved in file: %s" % filename)
                break

            tStopEpoch = time.time()
            print ("  Epoch Time Cost:", round(tStopEpoch - tStartEpoch,2), "s")
