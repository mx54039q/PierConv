#coding:utf-8
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

from ops import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784],'x')
y_actual = tf.placeholder(tf.float32, [None, 10],'label')

#Network
bn0 = batch_norm(name='bn0')
bn1 = batch_norm(name='bn1')
bn1_ = batch_norm(name='bn1_')
bn2 = batch_norm(name='bn2')
bn2_ = batch_norm(name='bn2_')

x_image = tf.reshape(x, [-1,28,28,1]) 
p0 = bn0(tf.layers.conv2d(x_image,
                        filters=8,
                        kernel_size=3,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                        name='conv0'))
p0_pr = tf.nn.relu(tf.layers.max_pooling2d(p0,2,2))
p1 = bn1(pure(p0_pr,16,'pure1'))
p1_ = bn1_(conv2d(p1,16,'conv1',kernel_size=1))
p1_pr = tf.nn.relu(tf.layers.max_pooling2d(p1_,2,2))
p2 = pure(p1_pr,32,'pure2')
p2_ = bn2_(conv2d(p2,32,'conv2',kernel_size=1))
size,channel = int(p2_.shape[1]),int(p2_.shape[3])
avg_reshape = tf.reshape(p2_,[-1,size*size*channel])
dense0 = tf.nn.relu(linear(avg_reshape,128,'dense0'))
logits = linear(dense0,10,'dense1')

y_predict = tf.nn.softmax(logits)


with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_actual,logits))
    
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess=tf.InteractiveSession()          
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
for i in range(5000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_actual: batch[1]})
    if i%100 == 0:                  #训练100次，验证一次
        train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1]})
        print('step',i,'training accuracy',train_acc)
    if(i%500 == 0):
        test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})
        print("test accuracy",test_acc)
