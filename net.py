from keras.layers import ConvLSTM2D, MaxPool3D, BatchNormalization, MaxPool2D
from tensorflow.contrib.layers import batch_norm

def controlNet(inputs, targets, shape, dropoutVec, branchConfig, params, scopeName = 'controlNET'):
    """
        Get one image/sequence of images to predict control operations for controling the vehicle
        inputs: N Batch of M images in order
        shape: [BatchSize, SeqSize, FrameHeight, FrameWeight, Channels]
        phase: placeholder for training
        scopeName: TensorFlow Scope Name to separate nets in the graph
    """
    with tf.variable_scope(scopeName) as scope:
        with tf.name_scope("Network"):

            networkTensor = load_imitation_learning_network(inputs[0], inputs[1],
                                                  shape[1:3], dropoutVec)
            solverList = []
            lossList = []
            trainVars = tf.trainable_variables()
            for i in range(0, len(branchConfig)):
                with tf.name_scope("Branch_" + str(i)):
                    if branchConfig[i][0] == "Speed":
                        # we only use the image as input to speed prediction
                        contLoss = tf.reduce_mean(tf.square(tf.subtract(networkTensor[-1], targets[0]))) #+ tf.add_n([tf.nn.l2_loss(v) for v in trainVars]) * L2NormConst
                        contSolver = tf.train.AdamOptimizer(learning_rate=params[3], beta1=params[4], beta2=params[5]).minimize(contLoss)
                        solverList.append(contSolver)
                        lossList.append(contLoss)
                        # create a summary to monitor cost tensor
                        tf.summary.scalar("Speed_Loss", contLoss)
                    else:
                        contLoss = tf.reduce_mean(tf.square(tf.subtract(networkTensor[i], targets[1]))) #+ tf.add_n([tf.nn.l2_loss(v) for v in trainVars]) * L2NormConst
                        contSolver = tf.train.AdamOptimizer(learning_rate=params[3], beta1=params[4], beta2=params[5]).minimize(contLoss)
                        solverList.append(contSolver)
                        lossList.append(contLoss)
                        tf.summary.scalar("Control_Loss_Branch_"+str(i), contLoss)

        tensors = {
            'optimizers' : solverList,
            'losses' : lossList,
            'output' : networkTensor
        }
    return tensors


#params = [trainScratch, dropoutVec, image_cut, learningRate, beta1, beta2, num_images, iterNum, batchSize, valBatchSize, NseqVal, epochs, samplesPerEpoch, L2NormConst]
def Net(branchConfig, params, timeNumberFrames, prefSize=(128, 160, 3)):
    shapeInput = [None, prefSize[0], prefSize[1], prefSize[2]]
    inputImages = tf.placeholder("float", shape=[None, prefSize[0], prefSize[1],
                                                                 prefSize[2]], name="input_image")
    inputData = []
    inputData.append(tf.placeholder(tf.float32,
                                           shape=[None, 4], name="input_control"))
    inputData.append(tf.placeholder(tf.float32,
                                           shape=[None, 1], name="input_speed"))

    inputs = [inputImages,inputData]
    dout = tf.placeholder("float", shape=[len(params[1])])

    targetSpeed = tf.placeholder(tf.float32, shape=[None, 1], name="target_speed")
    targetController = tf.placeholder(tf.float32, shape=[None, 3], name="target_control")

    targets = [targetSpeed, targetController]

    print('Building ControlNet ...')
    controlOpTensors = controlNet(inputs, targets, shapeInput, dout, branchConfig, params, scopeName = 'controlNET')

    tensors = {
            'inputs' : inputs,
            'targets' : targets,
            'params' : params,
            'dropoutVec' : dout,
            'output' : controlOpTensors
        }
    return tensors # [ inputs['inputImages','inputData'], targets['targetSpeed', 'targetController'],  'params', dropoutVec', output[optimizers, losses, branchesOutputs] ]
