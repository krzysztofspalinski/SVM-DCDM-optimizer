import numpy as np

class Model:

	def __init__(self, params=None, C=1):
		self.params = params
		self.C = C
		if params is None:
			self.__initialize_parameters()

	def predict(self, X):
		pass

	def train(self, X, optimizer):
		pass

	def compute_gradient(self, X):
		pass

	def __initialize_parameters(self):
		pass
