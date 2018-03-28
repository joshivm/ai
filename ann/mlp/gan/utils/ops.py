import tensorflow as tf
import numpy as np

class ops:

    def __init__(self,dtype):
        self.dtype = dtype

    def placeholder(self,shape,name,dtype=None):
        if dtype is None:
            return tf.placeholder(self.dtype,shape,name)
        else:
            return tf.placeholder(dtype,shape,name)
    def variable(self,shape,init,p=0.01,name=None):
        if init=="normal":
            initializer = tf.random_normal(shape,dtype=self.dtype,stddev=p)
        elif init == "uniform":
            initializer = tf.random_uniform(shape,-p,p,dtype=self.dtype)
        elif init=="xavier":
            initializer = self.xavier(shape,self.xavier_std(shape[1]))
        elif init == "zeros":
            initializer = tf.zeros(shape,dtype=self.dtype)
        elif init == "ones":
            initializer = tf.ones(shape,dtype=self.dtype)
        else:
            raise ValueError("Unknown initilizer for tensorflow variable "+init)
        return tf.Variable(initializer,dtype=self.dtype,name=name)

    def sample_noise(self,shape,dist="uniform"):
        if dist == "uniform":
            d = np.random.uniform(size=shape).astype(np.float32)
        elif dist == "normal":
            d = np.random.normal(size=shape).astype(np.float32)
        else:
            raise ValueError("Unknown value for the probability distribution of noise "+dist)
        return d

    def conv2d(self,x,w_shape,b_shape,strides=1,padding="VALID",name="conv2d"):
        with tf.variable_scope(name):
            std = x.get_shape().as_list()[1] * x.get_shape().as_list()[2] * x.get_shape().as_list()[3]
            w = tf.get_variable("w",initializer=self.xavier(w_shape,self.xavier_std(std)))
            b = tf.get_variable("b",initializer=self.xavier(b_shape,self.xavier_std(std)))
            return self.add(tf.nn.conv2d(x,w,strides=(1,strides,strides,1),padding=padding,name=name) , b)

    def maxpool(self,x,ksize=1,strides=1,padding="VALID",name="maxpool"):
        return tf.nn.max_pool(x,(1,ksize,ksize,1),(1,strides,strides,1),padding,name=name)

    def avgpool(self,x,ksize=1,strides=1,padding="VALID",name="avgpool"):
        return tf.nn.avg_pool(x,(1,ksize,ksize,1),(1,strides,strides,1),padding,name=name)

    def flatten(self,x):
        return tf.contrib.layers.flatten(x)

    def dropout(self,x,prob):
        return tf.nn.dropout(x,prob)

    def dense(self,x,w_shape,b_shape,_x=False,_w=False,name="dense"):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable("w",initializer=self.xavier(w_shape,self.xavier_std(w_shape[0])))
                b = tf.get_variable("b",initializer=self.xavier(b_shape,self.xavier_std(w_shape[0])))
                return self.add(self.matmul(w,x,_w,_x),b)

    def batch_norm(self,x,pl_train,reuse,name):
        with tf.variable_scope(name) as scope:
            return tf.contrib.layers.batch_norm(x,is_training=pl_train,reuse=tf.AUTO_REUSE,scope=scope)

    def matmul(self,a,b,_a=False,_b=False,name="matmul"):
        return tf.matmul(a,b,transpose_a=_a,transpose_b=_b,name=name)

    def add(self,a,b,name="add"):
        return tf.add(a,b,name=name)

    def substract(self,a,b,name="sub"):
        return tf.substract(a,b,name=name)

    def mult(self,a,b,name="mult"):
        return tf.multiply(a,b,name=name)

    def errorfn(self,a,b,dim=0,efn="mse",name="errorfn"):
        e = None
        if efn == "mse":
            e = tf.losses.mean_squared_error(labels=a,predictions=b,name=name)
        elif efn == "categorical_crossentropy":
            e = tf.nn.softmax_cross_entropy_with_logits_v2(labels=a,logits=b,dim=dim,name=name)
        elif efn == "binary_crossentropy":
            e = tf.nn.sigmoid_cross_entropy_with_logits(labels=a,logits=b,name=name)
        else:
            raise ValueError("Unkown error function "+efn)
        return tf.reduce_mean(e)

    def activation(self,a,afn=None,name="activation"):
        if afn == "sigmoid":
            a = tf.nn.sigmoid(a,name=name)
        elif afn == "softmax":
            a = tf.nn.softmax(a,name=name)
        elif afn == "relu":
            a = tf.nn.relu(a,name=name)
        elif afn == "tanh":
            a = tf.nn.tanh(a,name=name)
        elif afn == "maxout":
            a = tf.contrib.layers.maxout(a[0],a[1],axis=0)
        else:
            raise ValueError("Unknown activation function "+afn)
        return a

    def xavier_std(self,x):
        return tf.sqrt(1/x)

    def xavier(self,shape,std):
        return tf.random_normal(shape,dtype=self.dtype,stddev=std)

    def optimizer(self,l,lr,ofn="SGD",var_list = None):
        o = None
        if ofn == "SGD":
            if var_list is None:
                o = tf.train.GradientDescentOptimizer(lr).minimize(l)
            else:
                o = tf.train.GradientDescentOptimizer(lr).minimize(l,var_list=var_list)
        if ofn == "Adam":
            if var_list is None:
                o = tf.train.AdamOptimizer(lr).minimize(l)
            else:
                o = tf.train.AdamOptimizer(lr).minimize(l,var_list=var_list)
        return o

    def batch_eval(self,sess,opfn,pl_x,pl_y,x,y,b_len=5000,pl_train=False):
        start = 0
        at = 0
        len_x = len(x)
        n_b = int(np.ceil(len_x/b_len))
        n_avg = len_x/b_len
        for i in range(n_b):
            last = len(x) if (start+b_len) > len(x) else (start+b_len)
            bn = {pl_x:x[start:last],pl_y:y[start:last].T,pl_train:False}
            a = sess.run(opfn,feed_dict=bn)
            at += a
            start = last
        return at/n_avg

    def parse(self,x):
        return '{0:.4f}'.format(x)
