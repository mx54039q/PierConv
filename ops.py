import tensorflow as tf


def pure(inputs, filters, name, kernel_size=3, strides=(1,1),padding='same'):
    with tf.variable_scope(name) as scope:
        last_channels = int(inputs.shape[3])
        inputs_split = tf.split(inputs,last_channels,3)
        outs = []
        for i in range(filters):
            for j in range(last_channels):
                name_conv = name + '_' + str(i*last_channels+j)
                temp = tf.layers.conv2d(inputs_split[j],
                                filters=1,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=True,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                name=name_conv)
                outs.append(temp)
        return tf.concat(outs,3)
                
def conv2d(inputs,filters,name,kernel_size=3,strides=(1,1),padding='same'):
    return tf.layers.conv2d(inputs,
                            filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                            name=name)
        
def linear(inputs, units,name):
    return tf.layers.dense(inputs=inputs, 
                        units=units,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                        name=name)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name
  def __call__(self, x, is_train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=is_train,
                      scope=self.name)


  
  
  
  
  
  
