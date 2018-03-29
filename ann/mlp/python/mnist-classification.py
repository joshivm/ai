import tensorflow as tf
import numpy as np

import utils.ops as ops
import utils.draw as draw
import utils.log as log

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../../../',one_hot=True)

op = ops(tf.float32)
plt = draw('./')
epochs = 30
batch_size = 128
############################### Network parameters
X = op.placeholder([None,784],"X")
Y = op.placeholder([None,10],"Y")
lr = op.exponential_decay(0.01,epochs,0.9)
w1,b1 = [784,256],[1,256]
w2,b2 = [256,128],[1,128]
w3,b3 = [128,10],[1,10]

############################### Graph

def mlp(x):

    z1 = op.dense(x,w1,b1,name="dense1")
    a1 = op.activation(z1,"sigmoid")

    z2 = op.dense(a1,w2,b2,name="dense2")
    a2 = op.activation(z2,"sigmoid")

    z3 = op.dense(a2,w3,b3)
    a3 = op.activation(z3,"softmax",name="dense3")

    return a3,z3

y,logits = mlp(X)

loss = op.errorfn(Y,y,efn="categorical_crossentropy")

optimizer = op.optimizer(loss,lr,ofn="Adam")

correct = tf.equal(tf.argmax(Y,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))*100

sess = tf.InteractiveSession()

init = tf.global_variables_initializer()

sess.run(init)

train_accuracy = list()
test_accuracy = list()

for epoch in range(epochs):
    for batch in range(mnist.train.num_examples//batch_size):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        feed = { X : batch_x, Y : batch_y }
        sess.run(optimizer,feed)
    tr = op.batch_eval(sess,accuracy,X,Y,mnist.train.images,mnist.train.labels)
    te = op.batch_eval(sess,accuracy,X,Y,mnist.test.images,mnist.test.labels)
    train_accuracy.append(tr)
    test_accuracy.append(te)

    print(op.parse(tr),op.parse(te))

plt.lines([range(epochs),range(epochs)],[train_accuracy,test_accuracy],["Train","Test"],["-*","-o"],
          title="Accuracy",xlab="Epochs",ylab="Accuracy",int_tick=True,name="mlp",format='png')
