import tensorflow as tf

class logging():

    def __init__(self,sess,path):
        self.sess = sess
        self.path = path

    def set(self,scalars,hists):
        self.scalar(scalars)
        self.hist(hists)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.path,self.sess.graph)

    def scalar(self,scalars):
        for s in scalars:
            tf.summary.scalar(s[1],s[0])

    def hist(self,hists):
        for h in hists:
            tf.summary.histogram(h[1],h[0])

    def add_summary(self,feed,j):
        summary = self.sess.run(self.merged,feed)
        self.add(summary,j)

    def add(self,sum,j):
        self.train_writer.add_summary(sum,j)
