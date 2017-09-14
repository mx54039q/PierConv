import tensorflow as tf 
import cifar10_input
from ops import *
from equal import *

def convnet(x_images,reuse=False):
    with tf.variable_scope("Model",reuse=reuse) as scope:
        #Network
        bn0 = batch_norm(name='bn0')
        bn1 = batch_norm(name='bn1')
        bn1_ = batch_norm(name='bn1_')
        bn2 = batch_norm(name='bn2')
        bn2_ = batch_norm(name='bn2_')
        p1 = bn0(tf.layers.conv2d(x_images,
                                filters=16,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                name='conv1'))
        p1_ = tf.nn.relu(tf.layers.max_pooling2d(p1,2,2))
        p2 = bn1(tf.layers.conv2d(p1_,
                                filters=32,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                name='conv21'))
        p2_ = bn1_(conv2d(p2,32,'conv22',kernel_size=1))
        p2_ = tf.nn.relu(tf.layers.max_pooling2d(p2,2,2))
        p3 = bn2(tf.layers.conv2d(p2_,
                                filters=64,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                name='conv31'))
        p3_ = bn2_(conv2d(p2,64,'conv32',kernel_size=1))                        
        size,channel = int(p3.shape[1]),int(p3.shape[3])
        avg_reshape = tf.reshape(p3,[-1,size*size*channel])
        dense0 = linear(avg_reshape,128,'dense0')
        logits = linear(dense0,10,'dense1')
        return logits
        
def purenet(x_images,reuse=False):
    with tf.variable_scope("PureNet",reuse=reuse) as scope:
        bn0 = batch_norm(name='bn0')
        bn1 = batch_norm(name='bn1')
        bn1_ = batch_norm(name='bn1_')
        bn2 = batch_norm(name='bn2')
        bn2_ = batch_norm(name='bn2_')
        p0 = bn0(conv2d(x_images,16,'conv0',kernel_size=3))
        p0_pr = tf.nn.relu(tf.layers.max_pooling2d(p0,2,2))
        p1 = bn1(equal(p0_pr,4,'equal1'))
        p1_ = bn1_(conv2d(p1,32,'conv1',kernel_size=1))
        p1_pr = tf.nn.relu(tf.layers.max_pooling2d(p1_,2,2))
        p2 = bn2(equal(p1_pr,4,'equal'))
        p2_ = bn2_(conv2d(p2,64,'conv2',kernel_size=1))
        size,channel = int(p2_.shape[1]),int(p2_.shape[3])
        avg_reshape = tf.reshape(p2_,[-1,size*size*channel])
        dense0 = tf.nn.relu(linear(avg_reshape,128,'dense0'))
        logits = linear(dense0,10,'dense1')
        return logits

def equalnet(x_images,reuse=False):
    with tf.variable_scope("EqualNet",reuse=reuse) as scope:
        bn0 = batch_norm(name='bn0')
        bn0_ = batch_norm(name='bn0_')
        bn1 = batch_norm(name='bn1')
        bn1_ = batch_norm(name='bn1_')
        bn2 = batch_norm(name='bn2')
        bn2_ = batch_norm(name='bn2_')
        p0 = bn0(EqualImp(x_images,16,'equal0'))
        #p0_ = bn0_(conv2d(p0,4,'conv00',kernel_size=1))
        p0_pr = tf.nn.relu(tf.layers.max_pooling2d(p0,2,2))
        p1 = bn1(EqualImp(p0_pr,32,'equal1'))
        #p1_ = bn1_(conv2d(p1,32,'conv1',kernel_size=1))
        p1_pr = tf.nn.relu(tf.layers.max_pooling2d(p1,2,2))
        p2 = bn2(EqualImp(p1_pr,64,'equal2'))
        #p2_ = bn2_(conv2d(p2,64,'conv2',kernel_size=1))
        size,channel = int(p2.shape[1]),int(p2.shape[3])
        avg_reshape = tf.reshape(p2,[-1,size*size*channel])
        dense0 = tf.nn.relu(linear(avg_reshape,128,'dense0'))
        logits = linear(dense0,10,'dense1')
        return logits
        
def pressnet(x_images,reuse=False):
    with tf.variable_scope("PressNet",reuse=reuse) as scope:
        bn0 = batch_norm(name='bn0')
        bn0_ = batch_norm(name='bn0_')
        bn1 = batch_norm(name='bn1')
        bn1_ = batch_norm(name='bn1_')
        bn2 = batch_norm(name='bn2')
        bn2_ = batch_norm(name='bn2_')
        p0 = bn0(conv2d(x_images,1,'conv00',kernel_size=1))
        p0_ = bn0_(conv2d(p0,16,'conv01',kernel_size=3))
        p0_pr = tf.nn.relu(tf.layers.max_pooling2d(p0_,2,2))
        p1 = bn1(conv2d(p0_pr,1,'conv10',kernel_size=1))
        p1_ = bn1_(conv2d(p1,32,'conv11',kernel_size=3))
        p1_pr = tf.nn.relu(tf.layers.max_pooling2d(p1_,2,2))
        p2 = bn2(conv2d(p1_pr,1,'conv20',kernel_size=1))
        p2_ = bn2_(conv2d(p2,64,'conv21',kernel_size=3))
        size,channel = int(p2_.shape[1]),int(p2_.shape[3])
        avg_reshape = tf.reshape(p2_,[-1,size*size*channel])
        dense0 = tf.nn.relu(linear(avg_reshape,128,'dense0'))
        logits = linear(dense0,10,'dense1')
        return logits        
        
