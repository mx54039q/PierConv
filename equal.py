import tensorflow as tf
from tensorflow.python.ops import math_ops
from ops import *

def equal(inputs, filters, name, kernel_size=3, strides=(1,1),padding='same'):
    """
    channel-wise special convolution with the same kernel,
    output (input channels*filters) channels. 
    """
    last_channels = int(inputs.shape[3])
    inputs_split = tf.split(inputs,last_channels,3)
    outs = []
    for i in range(filters):
        name_conv = name + '_' + str(i)
        temp = tf.layers.conv2d(inputs_split[0],
                                filters=1,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                name=name_conv)
        outs.append(temp)
        for j in range(1,last_channels):
            temp = tf.layers.conv2d(inputs_split[j],
                                filters=1,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                name=name_conv,
                                reuse=True)
            outs.append(temp)
    return tf.concat(outs,3)
    
def Equal2D(inputs,
           filters,
           name,
           kernel_size=3,
           strides=1,
           use_bias=True,
           padding='valid',
           kernel_regularizer=None):
    """
    channel-wise special convolution with the same kernel,
    output (input channels*filters) channels. 
    """
    with tf.variable_scope(name) as scope:
        input_shape = inputs.shape.as_list()
        input_row, input_col = input_shape[1:3]
        input_filter = input_shape[3]
        
        output_row = input_row - kernel_size + 1
        output_col = input_col - kernel_size + 1
        kernel_shape = (1,kernel_size * kernel_size, filters)###
        kernel = tf.get_variable('kernel',kernel_shape,
                                initializer=tf.truncated_normal_initializer(stddev=0.01),
                                regularizer=tf.contrib.layers.l2_regularizer(0.003))
        if use_bias:
          bias = tf.get_variable('bias',(1,1,filters),
                                initializer=tf.zeros_initializer)
        
        stride_row = stride_col = strides
        _, feature_dim, filters = kernel_shape
        
        xs = []
        for k in range(input_filter):
            for i in range(output_row):
              for j in range(output_col):
                slice_row = slice(i * stride_row,
                                  i * stride_row + kernel_size)
                slice_col = slice(j * stride_col,
                                  j * stride_col + kernel_size)
                xs.append(
                    tf.reshape(inputs[:, slice_row, slice_col, k], (-1, 1, feature_dim
                                                                  )))
        x_aggregate = tf.concat(xs, axis=1) # x_aggregate(100,30*30*8,3*3)
        x_aggregate = tf.reshape(x_aggregate, (1,-1,feature_dim))
        output = batch_dot(x_aggregate, kernel) # kernel:(1,3*3,16)
        if use_bias:
            output += bias
        output = tf.reshape(output, (-1,output_row, output_col,
                                    filters*input_filter))
        
        #output = K.permute_dimensions(output, (2, 0, 1, 3))
        return output

  
def batch_dot(x, y, axes=None):
    return math_ops.matmul(x, y, adjoint_a=None, adjoint_b=None)

def EqualImp(inputs, filters, name, kernel_size=3, strides=(1,1),padding='same'):
    """
    channel-wise special convolution with the same kernel, and then 
    apply 1x1 convolution to 1 channnel, output size of filters channels. 
    """
    with tf.variable_scope(name) as scope:
        last_channels = int(inputs.shape[3])
        inputs_split = tf.split(inputs,last_channels,3)
        outs = []
        for i in range(filters):
            outs_channels = []
            name_conv = name + '_' + str(i)
            temp = tf.layers.conv2d(inputs_split[0],
                                    filters=1,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=True,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                    name=name_conv)
            outs_channels.append(temp)
            for j in range(1,last_channels):
                temp = tf.layers.conv2d(inputs_split[j],
                                    filters=1,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=True,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                    name=name_conv,
                                    reuse=True)
                outs_channels.append(temp)
            outs.append(conv2d(tf.concat(outs_channels,3),1,name_conv+'_press',kernel_size=1))
        return tf.concat(outs,3)
        
def EqualDiv(inputs, filters, name, kernel_size=3, strides=(1,1),padding='same'):
    """
    Divide inputs into 2 half, and then channel-wise special convolution 
    with the same kernel, and then apply 1x1 convolution to 1 channnel,
    output size of filters channels. 
    """
    with tf.variable_scope(name) as scope:
        last_channels = int(inputs.shape[3])
        inputs_split = tf.split(inputs,last_channels,3)
        outs = []
        for i in range(filters):
            outs_channels = []
            name_conv = name + '_' + str(i)
            temp = tf.layers.conv2d(inputs_split[0],
                                    filters=1,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=True,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                    name=name_conv)
            outs_channels.append(temp)
            for j in range(1,last_channels):
                temp = tf.layers.conv2d(inputs_split[j],
                                    filters=1,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    use_bias=True,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                    name=name_conv,
                                    reuse=True)
                outs_channels.append(temp)
            outs.append(conv2d(tf.concat(outs_channels,3),1,name_conv+'_press',kernel_size=1))
        return tf.concat(outs,3)        
        
        
        
