"""
Goodfellow et. al. 2014, description - https://github.com/goodfeli/adversarial/blob/master/mnist.yaml
- Use only 50,000 samples from the MNIST dataset
- Generator network:
    Sample from uniform distribution
    layers : [100,1200,1200,784]
    activation:[None,Relu,Relu,sigmoid]
    w_init : [-0.05,0.05]

- Discriminator network:
    layers: [784,240,240,1]
    dropout:[0.5,0.8] - first try network without dropouts
    scale: [2.,1.25] - try scaling up and scaling down
    activation: [None,Maxout,Maxout,sigmoid]
    w_init: [-0.005,0.005]
    num_pieces = 5,5
- batch_size = 100
- learning_rate = 0.1
- learning_rule = Momentum, init = 0.5
- Exponential decay with factor of 1.000004 and min lr = .000001
- Momentum adjustor: start 1, saturate 250, final 0.7
"""
################### Ignore future warning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
################### Required imports
import tensorflow as tf
import numpy as np
from utils.ops import ops
from utils.draw import draw
from utils.log import logging
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./");

path = "./figs/"
op = ops(tf.float32)
plt = draw(path=path,size=25)
############################ Input
epochs = 30
batch_size = 100
inp_range = 500
tf.set_random_seed(80)
############################ Network params

G_step = tf.Variable(0, trainable=False)
G_lr_s = 0.1
G_lr = tf.train.exponential_decay(G_lr_s, G_step,epochs*inp_range, 0.96, staircase=True)

D_step = tf.Variable(0, trainable=False)
D_lr_s = 0.01
D_lr = tf.train.exponential_decay(D_lr_s, D_step,epochs*inp_range, 0.96, staircase=True)


G_layers = [100,1200,1200,784]
D_layers = [784,240,240,1]

X = op.placeholder([G_layers[-1],batch_size],"X")
Y = op.placeholder([D_layers[-1],batch_size],"Y")
Z = op.placeholder([G_layers[0],batch_size],"Z")
pl_train = op.placeholder([],"train_status",tf.bool)
reuse = op.placeholder([],"reuse_status",tf.bool)
# D_lr = op.placeholder([],"D_learning_rate")
# G_lr = op.placeholder([],"G_learning_rate")


Gw1 = op.variable([G_layers[1],G_layers[0]],"xavier",0.05,"Gw1")
Gb1 = op.variable([G_layers[1],1],"zeros","Gb1")

Gw2 = op.variable([G_layers[2],G_layers[1]],"xavier",0.05,"Gw2")
Gb2 = op.variable([G_layers[2],1],"zeros","Gb2")

Gw3 = op.variable([G_layers[3],G_layers[2]],"xavier",0.05,"Gw3")
Gb3 = op.variable([G_layers[3],1],"zeros","Gb3")
G_list = [Gw1,Gb1,Gw2,Gb2,Gw3,Gb3]

Dw1 = op.variable([D_layers[1],D_layers[0]],"xavier",0.005,"Dw1")
Db1 = op.variable([D_layers[1],1],"zeros","Db1")

Dw2 = op.variable([D_layers[2],D_layers[1]],"xavier",0.005,"Dw2")
Db2 = op.variable([D_layers[2],1],"zeros","Db2")

Dw3 = op.variable([D_layers[3],D_layers[2]],"xavier",0.005,"Dw3")
Db3 = op.variable([D_layers[3],1],"zeros","Db3")
D_list = [Dw1,Db1,Dw2,Db2,Dw3,Db3]
########################### Graph

def G(z):
    gz1 = op.matmul(Gw1,z)+Gb1
    ga1 = op.activation(gz1,"relu")

    ga1_b = op.batch_norm(ga1,pl_train,reuse,"gbatch1")

    gz2 = op.matmul(Gw2,ga1_b) +Gb2
    ga2 = op.activation(gz2,"relu")

    ga2_b = op.batch_norm(ga2,pl_train,reuse,"gbatch2")

    gz3 = op.matmul(Gw3,ga2_b) +Gb3
    ga3 = op.activation(gz3,"sigmoid")
    return ga3

def D(x):
    # x_d = op.dropout(x,1.)

    dz1 = op.matmul(Dw1,x) +Db1
    da1 = op.activation(dz1,"relu")
    # da1_d = op.dropout(da1,0.8)

    dz2 = op.matmul(Dw2,da1) +Db2
    da2 = op.activation(dz2,"relu")

    dz3 = op.matmul(Dw3,da2) +Db3
    da3 = op.activation(dz3,"sigmoid")
    return da3,dz3

########################### Objective
D_fake,D_logits_fake = D(G(Z))
D_real,D_logits_real = D(X)

real_ones = tf.ones_like(D_real,dtype=tf.float32)
fake_zeros = tf.zeros_like(D_fake,dtype=tf.float32)
fake_ones = tf.ones_like(D_fake,dtype=tf.float32)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_ones,logits=D_logits_real))
                #op.errorfn(real_ones,D_real,efn="categorical_crossentropy")
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_zeros,logits=D_logits_fake))
                #op.errorfn(real_zeros,D_fake,efn="categorical_crossentropy")

D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_ones,logits=D_logits_fake))
            #op.errorfn(fake_ones,D_fake,efn="categorical_crossentropy")

D_optimizer = tf.train.MomentumOptimizer(D_lr,0.5).minimize(D_loss,var_list=D_list)
                #op.optimizer(D_loss,D_lr,"Adam",var_list=D_list)
G_optimizer = tf.train.MomentumOptimizer(G_lr,0.5).minimize(G_loss,var_list=G_list)
                #op.optimizer(G_loss,G_lr,"Adam",var_list=G_list)


############################ Training
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

############################ Logging
logs = logging(sess,"./")

scalars = [[D_loss,'D_loss'],[G_loss,'G_loss']]
hists = [
    [Gw1,'Gw1'],[Gb1,'Gb1'],[Gw2,'Gw2'],[Gb2,'Gb2'],[Gw3,'Gw3'],[Gb3,'Gb3'],
        [Dw1,'Dw1'],[Db1,'Db1'],[Dw2,'Dw2'],[Db2,'Db2'],[Dw3,'Dw3'],[Db3,'Db3'],
        [D_real,'D_real'],[D_fake,'D_fake']
]
logs.set(scalars,hists)

############################


dloss = list()
gloss = list()
for i in range(epochs):
    for j in range(inp_range):
        bn = mnist.train.next_batch(batch_size)
        feedD = {X:bn[0].T,Z:op.sample_noise([G_layers[0],batch_size],"normal"),pl_train:True,reuse:False}
        dl,_ = sess.run([D_loss,D_optimizer],feed_dict=feedD)
        feedG = {Z:op.sample_noise([G_layers[0],batch_size],"normal"),pl_train:True,reuse:False}
        gl,_ = sess.run([G_loss,G_optimizer],feed_dict=feedG)
    logs.add_summary(feedD,i)
    print("EP:"+str(i)," ",op.parse(dl)," ",op.parse(gl))
    dloss.append(dl)
    gloss.append(gl)

pred = sess.run(G(Z),feed_dict={Z:op.sample_noise([G_layers[0],100],"normal"),pl_train:False,reuse:True})
plt.show_imgs(pred[:,0:16],"gray",name="gan-result")
plt.lines([range(epochs),range(epochs),range(epochs),range(epochs)],[dloss,gloss,2*0.693*np.ones(epochs),0.693*np.ones(epochs)],\
          ["D Loss","G Loss","1.386","0.693"],["-*","-o","--","--"],\
          "GAN MLP training loss","Epoch","Binary crossentropy loss",int_tick=True,loc="upper right",name="gan-loss")
