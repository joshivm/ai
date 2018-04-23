import tensorflow as tf
import numpy as np
from utils.ops import ops
from utils.draw import draw
from utils.log import logging
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/vinay/Dropbox/Vinay/vinay_work/datasets/mnist/",one_hot=True);
images = np.array(255*mnist.train.images[0:1000,:]).reshape(-1,28,28,1)

url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
sess = tf.InteractiveSession()

a = tf.contrib.gan.eval.run_inception(
    tf.contrib.gan.eval.preprocess_image(tf.constant(images)),
    default_graph_def_fn = tf.contrib.gan.eval.get_graph_def_from_url_tarball(url,"classify_image_graph_def.pb")
)

print(sess.run(a))
