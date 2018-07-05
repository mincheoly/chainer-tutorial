"""
	linear_models.py

	Some simple linear models to learn how to write my own loss function etc.
"""

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Define a forward pass function taking the data as input.
# and the linear function as output.
def linear_forward(data):
    return linear_function(data)


# Define a training function given the input data, target data,
# and number of epochs to train over.
def linear_train(train_data, train_target,n_epochs=200):

    for _ in range(n_epochs):
        # Get the result of the forward pass.    
        output = linear_forward(train_data)

        # Calculate the loss between the training data and target data.
        loss = F.mean_squared_error(train_target,output)

        # Zero all gradients before updating them.
        linear_function.zerograds()

        # Calculate and update all gradients.
        loss.backward()

        # Use the optmizer to move all parameters of the network
        # to values which will reduce the loss.
        optimizer.update()

if __name__ == '__main__':

	x = 30*np.random.rand(1000).astype(np.float32)
	y = 7*x+10
	y += 10*np.random.randn(1000).astype(np.float32)
	y = y > y.mean()

	# Setup linear link from one variable to another.

	linear_function = L.Linear(1,1)

	# Set x and y as chainer variables, make sure to reshape
	# them to give one value at a time.
	x_var = Variable(x.reshape(1000,-1))
	y_var = Variable(y.reshape(1000,-1).astype(np.int32), requires_grad=False)

	# Setup the optimizer.
	optimizer = optimizers.MomentumSGD(lr=0.001)
	optimizer.setup(linear_function)

	for i in range(100):

		# Get the result of the forward pass.    
		output = linear_forward(x_var)

		# Calculate the loss between the training data and target data.
		loss = F.sigmoid_cross_entropy(output, y_var)

		# Zero all gradients before updating them.
		linear_function.zerograds()

		# Calculate and update all gradients.
		loss.backward()

		# Use the optmizer to move all parameters of the network
		# to values which will reduce the loss.
		optimizer.update()

		print(loss)
