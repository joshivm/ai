Network.ipynb file is example implementation of the network class that is built in network.py. Any suggestions/comments/bug-reports are always welcome.

- [MLP](https://github.com/vinayjoshi22/ai/tree/master/ann/mlp) : Code contains python class for mlp. Network.ipynb is an example file demonstrating use of class network.py that implements mlp. Simple use for [MNIST dataset](https://pjreddie.com/projects/mnist-in-csv/) network is as follows,

Import network into the python code
```
import network as Network
```

Build the network with architecture desired
```
mnist = Network.Network(layer_units,
                        learning_rate,
                          epochs,
                          x_train,
                          y_train,
                          x_test,
                          y_test,
                          x_val,
                          y_val,
                          lrSet=None)
```
layer_units - list of number of units in each layer including input and output layers

learning_rate - initial learning_rate

epochs - number of epochs to train network with

x_train - input for train set

y_train - labels for train set

x_test - input for test set

y_test - labels for test set

x_val - input for validation set

y_val - labels for validation set

lrSet - list of learning_rate values, should be same as number of epochs

Train network - returns training progress of the network

```
mnist.train()
```

Test network - returns performance of the network on test set
```
mnist.test()
```

To access the parameters of the network
```
mnist.weights
mnist.bias
```
To reevaluate networks
```
mnist.accuracy(x_train,y_train) // returns train accuracy
```

Note: To run sample for MNIST you should download [MNIST dataset](https://pjreddie.com/projects/mnist-in-csv/) and put it in the same folder as [Network.ipynb](https://github.com/vinayjoshi22/ai/blob/master/ann/mlp/python/Network.ipynb). You can use iPython to run this interactively or directly in command prompt.
