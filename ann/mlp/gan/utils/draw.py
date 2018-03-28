import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

class draw:
    def __init__(self,path="./",family="sans-serif",weight="bold",size=20):
        font = {'family' : family,
                'weight' : weight,
                'size'   : size}
        params = {'axes.labelsize': size,
                  'axes.labelweight': weight,
                  'axes.titlesize':size,
                  'axes.titleweight':weight}
        plt.rc('font', **font)
        plt.rcParams.update(params)
        self.fig,self.ax= None,None
        self.fontsize = size
        self.path = path

    def line(self,x,y,title="",xlab="",ylab="",style="-*",lw=3.0,int_tick=False,w=12,h=10,name=None,format=None):
        self.fig,self.ax = plt.subplots()
        plt.plot(x,y,style,linewidth=lw)
        self.ax.set_xlabel(xlab,fontsize=self.fontsize)
        self.ax.set_ylabel(ylab,fontsize=self.fontsize)
        self.ax.set_title(title,fontsize=self.fontsize)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=int_tick))
        plt.grid(True)
        self.fig.set_size_inches(w, h, forward=True)
        if name:
            plt.savefig("{}.pdf".format(self.path+name),format="pdf")
            print("Saving figure at "+self.path+name+".pdf")
            if format:
                plt.savefig(name,format=format)
        else:
            plt.show()

    def lines(self,xs,ys,labels,styles,title="",xlab="",ylab="",lw=3.0,
              int_tick=False,w=12,h=10,loc="lower right",name=None,format=None):
        self.fig,self.ax = plt.subplots()
        for x,y,label,style in zip(xs,ys,labels,styles):
            plt.plot(x,y,style,label=label,linewidth=lw)
        self.ax.legend(loc=loc)
        self.ax.set_xlabel(xlab,fontsize=self.fontsize)
        self.ax.set_ylabel(ylab,fontsize=self.fontsize)
        self.ax.set_title(title,fontsize=self.fontsize)
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=int_tick))
        plt.grid(True)
        self.fig.set_size_inches(w, h, forward=True)
        if name:
            plt.savefig("{}.pdf".format(self.path+name),format="pdf")
            print("Saving figure at "+self.path+name+".pdf")
            if format:
                plt.savefig(name,format=format)
        else:
            plt.show()

    def show_imgs(self,pred,cmap,name=None,format=None):
        sbp = 4 # int(np.sqrt(pred.shape[-1]))
        img_sh = 28 # int(np.sqrt(pred.shape[0]))
        self.fig,self.ax = plt.subplots(sbp,sbp)
        for i in range(sbp):
            for j in range(sbp):
                img = pred[:,sbp*i+j].reshape(img_sh,img_sh)
                self.ax[i,j].imshow(img,cmap=cmap)
                self.ax[i,j].set_xticklabels([])
                self.ax[i,j].set_yticklabels([])
        if name:
            plt.savefig("{}.pdf".format(self.path+name),format="pdf")
            if format:
                plt.savefig(name,format=format)
        else:
            plt.show()
