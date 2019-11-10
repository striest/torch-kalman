"""
Basically will just contain funtions to build the transition and control matrices for various systems.
"""

import torch

class XVAAccControlMatrixGen:
	"""
	Contructor class for state [x, v, a], control over acceleration. Note that these functions are supposed to work in batch.
	"""

	def __init__(self, dt=0.1):
		self.dt = dt

	def build_F(self, state):
		batch_size = state.shape[0]
		return torch.tensor([[1, self.dt, 0.5*(self.dt**2)], [0, 1, self.dt], [0, 0, 0]]).repeat(batch_size, 1, 1)

	def build_B(self, control):
		batch_size = control.shape[0]
		return torch.tensor([[0.5*(self.dt**2)], [self.dt], [1.]]).repeat(batch_size, 1, 1)

	def build_H(self, state):
		batch_size = state.shape[0]
		return torch.eye(3).repeat(batch_size, 1, 1)