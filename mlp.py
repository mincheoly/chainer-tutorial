import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist

class MLP(chainer.Chain):

	def __init__(self, n_mid_units=100, n_out=10):

		super(MLP, self).__init__()

		with self.init_scope():

			self.l1 = L.Linear(None, n_mid_units)
			self.l2 = L.Linear(None, n_mid_units)
			self.l3 = L.Linear(None, n_mid_units)


	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		return F.relu(self.l3(hs))

if __name__ == '__main__':

	# Get the training and test data

	train, test = mnist.get_mnist()

	max_epoch = 10

	model = L.Classifier(MLP())

	optimizer = optimizers.MomentumSGD()

	optimizer.setup(model)

	updater = training.updaters.StandardUpdater(train_iter, optimizer, device=-1)