"""
Basically will just contain funtions to build the transition and control matrices for various systems.
"""

import torch
import matplotlib.pyplot as plt
import math

class XVAAccControlMatrixGen:
	"""
	Contructor class for state [x, v, a], control over acceleration. Note that these functions are supposed to work in batch.

	Note: expects state as a [batchx3x1] tensor, and control as [batchx1x1]
	"""

	def __init__(self, dt=0.1):
		self.dt = dt

	def build_F(self, state, control):
		batch_size = state.shape[0]
		return torch.tensor([[1, self.dt, 0.5*(self.dt**2)], [0, 1, self.dt], [0, 0, 0]]).repeat(batch_size, 1, 1)

	def build_B(self, state, control):
		batch_size = control.shape[0]
		return torch.tensor([[0.5*(self.dt**2)], [self.dt], [1.]]).repeat(batch_size, 1, 1)

	def build_H(self, state, control):
		batch_size = state.shape[0]
		return torch.eye(3).repeat(batch_size, 1, 1)

class XYVATJerkSteerControlGen:
	"""
	Constructor class for state = [x, y, v, a, theta] and control [a', theta']
	Note: expects state as a [batchx5x1] tensor, and control as [batchx2x1]
	"""
	def __init__(self, dt):
		self.dt = dt

	def build_F(self, state, control):
		"""
		In this scenario, F = 
		[[1    0    cos(tk)dt    0.5*cos(tk)dt^2    0]
		 [0    1    sin(tk)dt    0.5*sin(tk)dt^2    0]
		 [0    0    1            dt                 0]
		 [0    0    0            1                  0]
		 [0    0    0            0                  1]]
		"""
		dt = self.dt
		theta_k = state[:, 4, 0] + control[:, 1, 0]
		batch_size = state.shape[0]

		F = torch.tensor([	[1., 0., dt, 0.5*dt**2, 0.],
							[0., 1., dt, 0.5*dt**2, 0.],
							[0., 0., 1., dt, 0.], 
							[0., 0., 0., 1., 0],
							[0., 0., 0., 0., 1.]])
		F = F.repeat(batch_size, 1, 1)
		F[:, 0, 2] *= torch.cos(theta_k)
		F[:, 0, 3] *= torch.cos(theta_k)
		F[:, 1, 2] *= torch.sin(theta_k)
		F[:, 1, 3] *= torch.sin(theta_k)

		return F

	def build_B(self, state, control):
		dt = self.dt
		theta_k = state[:, 4, 0] + control[:, 1, 0]
		batch_size = control.shape[0]

		B = torch.tensor([[0.5*dt**3, 0.], [0.5*dt**3, 0.], [dt**2, 0.], [dt, 0.], [0., dt]])
		B = B.repeat(batch_size, 1, 1)
		B[:, 0, 0] *= torch.cos(theta_k)
		B[:, 1, 0] *= torch.sin(theta_k)

		return B

	def build_H(self, state, control):
		dt = self.dt
		batch_size = state.shape[0]
		return torch.eye(5).repeat(batch_size, 1, 1)

if __name__ == '__main__':
	"""
	Testing my kinematics for the 5DOF case.
	"""

	n_steps = 2000
	dt = 0.1
	n_feats = 5

	s_acc = []
	s = torch.zeros(1, n_feats, 1)
	u = torch.tensor([0.1, 0.1]).unsqueeze(0).unsqueeze(2)
	uf = lambda t: torch.tensor([.01 if t < 10 else (-.1 if t == 11 else 0.0), 0.13 if (t+200) % 800 < 400 else -0.13]).unsqueeze(0).unsqueeze(2)
	gen = XYVATJerkSteerControlGen(dt)

	for t in range(n_steps):
		u = uf(t)
		F = gen.build_F(s, u)
		B = gen.build_B(s, u)

		print('s = {}'.format(s))

		s = torch.bmm(F, s) + torch.bmm(B, u)

		print('s new = {}'.format(s))

		s_acc.append(s)

	#index as [batch, timestep, state_var]
	s = torch.stack(s_acc, dim=1).squeeze(3).numpy()
	t = torch.arange(0, n_steps*dt, dt).numpy()

	print(t.shape, s.shape)

	fig, ax = plt.subplots(n_feats+1, figsize=(4, 9))

	for f in range(n_feats):
		ax[f].scatter(t, s[0, :, f], s=0.5, c='b')

	#Also plot x,y
	ax[n_feats].plot(s[0, :, 0], s[0, :, 1], color='r')

	plt.show()










