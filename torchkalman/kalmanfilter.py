import torch
from torch import distributions
import numpy as np
import matplotlib.pyplot as plt

from torchkalman.utils import XVAAccControlMatrixGen

class KalmanFilter():
	"""
	Implementation of a Kalman filter in pytorch for trajectory smoothing. Note that we're using EKF(I think), 
	so instead of passing in the transition and control matrices, we pass in functions that construct them, given state/control signals.
	"""
	def __init__(self, state_dim, obs_dim, transition_m_f, control_m_f, obs_m_f, process_cov, obs_cov):
		self.state_dim = state_dim
		self.obs_dim = obs_dim
		self.transition_m_f = transition_m_f
		self.control_m_f = control_m_f
		self.obs_m_f = obs_m_f
		self.process_cov = process_cov
		self.obs_cov = obs_cov

	def smooth(self, trajs, control):
		"""
		Smooths a batch of trajectories. Expects trajectories to be batched as [batch_dim, timestep, state_var]
		We also assume first input into the trajectories is the initial state.
		"""
		n_trajs = trajs.shape[0]
		n_steps = trajs.shape[1]
		s = trajs[:, 0, :]
		u = control[:, 0, :]
		o = s.clone()

		state_acc = [s]

		#I think you can assume inital variance is 0.

		P = self.process_cov.repeat(n_trajs, 1, 1)
		Q = self.obs_cov.repeat(n_trajs, 1, 1)

		p_cov = P.clone()
		obs_cov = Q.clone()

		for t in range(1, n_steps):
			print('_'*30, 'T = {}'.format(t), '_'*30)

			u = control[:, t, :]

			F = self.transition_m_f(s, u)
			B = self.control_m_f(s, u)
			H = self.obs_m_f(s, u)

			print('F = {}'.format(F))
			print('B = {}'.format(B))
			print('H = {}'.format(H))
			print('P = {}'.format(P))
			print('Q = {}'.format(Q))

			obs = trajs[:, t, :]

			print('s_in = {}'.format(s))
			print('control = {}'.format(u))
			print('process cov = {}'.format(p_cov))
			#Get a-priori estimate of next state and covariance
			s_pred = torch.bmm(F, s) 
			s_pred += torch.bmm(B, u)
			p_cov_pred = torch.bmm(F, torch.bmm(p_cov, F.transpose(1, 2))) + p_cov


			print('obs = {}'.format(obs))
			print('s_pred = {}'.format(s_pred))
			print('process cov pred = {}'.format(p_cov_pred))

			#Incorporate measurements to refine a-priori estimate
			measurement_residual = obs - torch.bmm(H, s_pred)
			measurement_residual_cov = torch.bmm(H, torch.bmm(p_cov_pred, H.transpose(1, 2))) + obs_cov

			print('measurement error = {}'.format(measurement_residual))
			print('measurement cov error = {}'.format(measurement_residual_cov))

			kalman_gain = torch.bmm(p_cov_pred, torch.bmm(H.transpose(1, 2), measurement_residual_cov.inverse()))

			print('Kalman gain = {}'.format(kalman_gain))

			s = s_pred + torch.bmm(kalman_gain, measurement_residual)
			p_cov = torch.bmm((torch.eye(self.state_dim).repeat(n_trajs, 1, 1) - torch.bmm(kalman_gain,H)), p_cov_pred)

			print('s hat = {}'.format(s))
			print('p cov hat = {}'.format(p_cov))

			state_acc.append(s)

		out = torch.stack(state_acc, dim=1)
		return out




def generate_noisy_trajectories(n_trajs, n_steps, lambda_acc, process_std, obs_std, dt=0.1):
	"""
	Generates a batch of 1-D trajectories from 0 with process and observation noise as specified.
	n_trajs: amount of trajectories to generate
	n_steps: amount of timesteps in the trajectory
	lambda_acc: Func for un-noisy acceleration at timestep t.
	process_std: amount of noise to add to the control signal. (propagates to v, x)
	obs_std: amount of noise to add to the observation. (does not propagate to v, x)
	"""

	states_acc = []
	obses_acc = []

	#State is position, velocity, acc.
	init_states = torch.zeros(n_trajs, 3)

	#
	transition_m = torch.tensor([[1, dt, 0.5*(dt**2)], [0, 1, dt], [0, 0, 0]]).repeat(n_trajs, 1, 1)
	control_m = torch.tensor([[0.5*(dt**2)], [dt], [1]]).repeat(n_trajs, 1, 1)
	obs_m = torch.eye(3).repeat(n_trajs, 1, 1)

	states = init_states.clone().unsqueeze(2)
	states_acc.append(states)
	obses_acc.append(states)

	obs_noise = distributions.MultivariateNormal(torch.zeros(3), torch.diag(torch.tensor([obs_std]*3)))
	process_noise = distributions.MultivariateNormal(torch.zeros(3), torch.diag(torch.tensor([0.5*(dt**2), dt, process_std])))

	for t in range(n_steps):
		u = torch.tensor(lambda_acc(t)).repeat(n_trajs).unsqueeze(1).unsqueeze(2)
		print('u={}'.format(u))
		states = torch.bmm(transition_m, states) 
		states += torch.bmm(control_m, u)
		states += process_noise.sample([n_trajs]).t()
		states_acc.append(states)

		obses = torch.bmm(obs_m, states)
		obses += obs_noise.sample([n_trajs]).t()
		obses_acc.append(obses)

	states_out = torch.stack(states_acc, dim=1)
	obses_out = torch.stack(obses_acc, dim=1)

	return states_out, obses_out

if __name__ == '__main__':
	n_trajs = 1
	n_steps = 50
	acc = lambda t:min(max((t-15.) * 0.05, 0.), 1.)
	dt = 0.1

	process_std = 0.001
	obs_std = 0.01

	true_trajs, trajs = generate_noisy_trajectories(n_trajs, n_steps, acc, process_std, obs_std, dt)
	control = true_trajs[:, :, 2].unsqueeze(2)
	t = np.arange(0, n_steps*dt+dt, dt)
	x = trajs[0, :, 0]
	v = trajs[0, :, 1]
	a = trajs[0, :, 2]

	gen = XVAAccControlMatrixGen(dt)

	process_cov = torch.diag(torch.tensor([0.5*(dt**2), dt, process_std]))
	process_cov = torch.diag(torch.tensor([process_std]*3))
	obs_cov = torch.diag(torch.tensor([obs_std*100, obs_std*10, obs_std])) * 10

	kf = KalmanFilter(3, 3, gen.build_F, gen.build_B, gen.build_H, process_cov, obs_cov)

	print(trajs)

	s_trajs = kf.smooth(trajs, control)

	print(s_trajs)

	xs = s_trajs[0, :, 0]
	xv = s_trajs[0, :, 1]
	xa = s_trajs[0, :, 2]

	fig, ax= plt.subplots(3, 2, figsize=(8, 9))

	ax[0, 0].scatter(t, x, s=0.5)
	ax[1, 0].scatter(t, v, s=0.5)
	ax[2, 0].scatter(t, a, s=0.5)

	ax[0, 1].scatter(t, xs, s=0.5, c='r')
	ax[1, 1].scatter(t, xv, s=0.5, c='r')
	ax[2, 1].scatter(t, xa, s=0.5, c='r')

	plt.show()











