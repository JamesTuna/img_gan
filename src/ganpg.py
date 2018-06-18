try:
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
except:
    print("Tensorflow Not Found")
    tf = None
import numpy as np
from scipy.misc import *
import skimage.measure as skm

def stochasticInputProducer(imgLoc, batchSize, randSampling, pixelSize = 8):
    with open(imgLoc) as f:
        lists=np.array(f.readlines())

    dataLength = len(lists)
    #randSampling = np.random.randint(dataLength, size=batchSize)
    batchList = lists[randSampling]
    for i in range(batchSize):
        tmp=imread(batchList[i][:-1], mode='L')/255 # -1 for eliminating '\n' and /255 for transforming RGB255 to 1; 2048*2048
        tmp=skm.block_reduce(tmp, (pixelSize, pixelSize), np.average) # block reduce to 256*256 for default
        tmp=np.expand_dims(tmp, axis=0) # (1,256,256)
        if i==0:
            batchData=tmp
        else:
            batchData=np.concatenate((batchData,tmp), axis=0)
    batchData=np.expand_dims(batchData, axis=3) # (batchSize,256,256,1)
    return batchData

def fwdGeneratorAE(net, is_training = True):

    with slim.arg_scope([slim.conv2d_transpose, slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.02),
                      biases_initializer=tf.constant_initializer(0),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={'is_training': is_training}):

        net1 = slim.conv2d(net, 16, [5,5], stride=2, padding='same', scope='g_conv7') #128
        net2 = slim.conv2d(net1, 64, [5,5], stride=2, padding='same', scope='g_conv6') #64
        net3 = slim.conv2d(net2, 128, [5,5], stride=2, padding='same', scope='g_conv5') #32
        net4 = slim.conv2d(net3, 512, [5,5], stride=2, padding='same', scope='g_conv4') #16

        net5 = slim.conv2d(net4, 1024, [5,5], stride=2, padding='same', scope='g_conv3') #8

        net6 = slim.conv2d_transpose(net5, 512, [5,5], stride=2, padding='same', scope='g_dconv3') #16
        net6 = tf.concat([net4,net6],3)

        net7 = slim.conv2d_transpose(net6, 128, [5,5], stride=2, padding='same', scope='g_dconv4') #32
        net7 = tf.concat([net3,net7],3)

        net8 = slim.conv2d_transpose(net7, 64, [5,5], stride=2, padding='same', scope='g_dconv5') #64
        net8 = tf.concat([net2,net8],3)

        net9 = slim.conv2d_transpose(net8, 16, [5,5], stride=2, padding='same', scope='g_dconv6') #128
        net9 = tf.concat([net1,net9],3)

        net10 = slim.conv2d_transpose(net9, 1, [5,5], stride=2, padding='same', scope='g_dconv7') #256

        smooth_kernal = tf.constant(1.0,shape=[5,5,1,1])
        net10 = tf.nn.conv2d(net10,smooth_kernal,[1,1,1,1],'SAME')
        net10 = net10/5

    return net10

def fwdDiscriminator(net, is_training = True, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        with slim.arg_scope([slim.conv2d],
                          #activation_fn=tf.contrib.keras.layers.LeakyReLU,
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.02),
                          biases_initializer=tf.constant_initializer(0),
                          normalizer_fn=slim.batch_norm,
                          normalizer_params={'is_training': is_training}):
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='d_conv1')
            net = slim.conv2d(net, 64, [3, 3], stride=2, scope='d_pool1') #128
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='d_conv2')
            net = slim.conv2d(net, 128, [3, 3], stride=2, scope='d_pool2')#64
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='d_conv3')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='d_pool3')#32
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='d_conv4')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='d_pool4')#16
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='d_conv5')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='d_pool5')#8
            net = slim.flatten(net)
            net = slim.fully_connected(net, 2048, scope='d_fc6')
            net = slim.dropout(net, 0.5, scope='d_dropout6', is_training=is_training)
            net = slim.fully_connected(net, 512, scope='d_fc7')
            net = slim.dropout(net, 0.5, scope='d_dropout7', is_training=is_training)
            net = slim.fully_connected(net, 2, activation_fn=None, scope='d_fc8')
    return net

