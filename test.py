#coding:utf-8
import tensorflow as tf 
import cifar10_input
from ops import *
import model
import equal

data_dir = 'cifar10_data/cifar-10-batches-bin/'
batch_size = 100
x_images,labels = cifar10_input.distorted_inputs(data_dir,batch_size)
onehot_labels = tf.one_hot(labels,10)
#Model Constructe
logits = model.purenet(x_images)
y_predict = tf.nn.softmax(logits)
# Model Test
images_test, labels_test = cifar10_input.inputs('test',data_dir,10000)
onehot_labels_test = tf.one_hot(labels_test,10)
logits_test = model.purenet(images_test,reuse=True)
y_predict_test = tf.nn.softmax(logits_test)
# Loss Set
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels,logits))
    loss_test = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels_test,logits_test))
tf.summary.scalar('loss_train', loss)
tf.summary.scalar('loss_test', loss_test)
# Optimazer and Some Ops
train_op1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
train_op2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.cast(labels,'int64'))
test_prediction = tf.equal(tf.argmax(y_predict_test,1), tf.cast(labels_test,'int64'))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
accuracy_test = tf.reduce_mean(tf.cast(test_prediction, "float"))
tf.summary.scalar('acc_test', accuracy_test)
# Creat Sess
sess=tf.InteractiveSession()          
sess.run(tf.global_variables_initializer())
# start populating the filename queue
tf.train.start_queue_runners(sess=sess)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
l = 0.0
acc = 0.0
lr = 0.0
# Train 
for i in range(15000):
    if(i%500 == 0):
        l, acc = sess.run([loss_test,accuracy_test])
        print;print('step %d, test accuracy: %.4f, test loss: %.4f' % (i,acc,l));print
    if(i < 5000):
        _, l, acc = sess.run([train_op1,loss,accuracy])
        lr = 0.001
    else:
        _, l, acc = sess.run([train_op2,loss,accuracy])
        lr = 0.0001
    if i%100 == 0:                  #训练100次，验证一次
        merge_sum = sess.run(merged)
        writer.add_summary(merge_sum, i)
        print('step %d, train accuracy: %.4f, train loss: %.4f, lr: %.4f' % (i,acc,l,lr))
    
        
        
        
        
        
        
        
        
        
        
        
        
