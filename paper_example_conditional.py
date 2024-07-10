#-------------------------------------------------------------------------------
# Name:        FFT-MA Level-Sim (Layered Copula)
# Purpose:     Simulation of non-Gaussian spatial random fields
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     01/07/2022, Centre for Natural Gas, EAIT,
#                          The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
from helper_func import fftma
import level_sim_conditional
import gstools as gs

# SIMULATE CONDITIONAL FIELDS
# read conditioning values
xy = np.load('data/small_xy.npy')
condfield = np.load('data/small_sample_field.npy')
cv = condfield[xy[:,0], xy[:,1]]

# plot field that is sampled 
plt.figure()
plt.imshow(condfield, interpolation='nearest', origin='lower', cmap='jet', vmin=-3.6, vmax=3.6)
plt.plot(xy[:,1], xy[:,0], 'x')
plt.colorbar()
plt.savefig(r'sampled_field.png')
plt.clf()
plt.close()

# number of conditional realizations
n_realisations = 10

cond_fields = []
for s in range(n_realisations):
	print('Simulate conditional realization # {}'.format(s))
	covmods = []
	nlev = 41 # has to be an odd number to get phi_(tau)=0
	tau = np.linspace(0.0, 1, nlev)
	r1 = 40
	r2 = 4
	for ii, r in enumerate(tau):
		r = (1- tau[ii])*r1 + tau[ii]*r2
		covmod =  '0.01 Nug(0.0) + 0.99 Exp({})'.format(r)
		covmods.append(covmod)
	print(covmods)

	# initialize FFTMA_LS with the covmodes defined above
	fftmals = level_sim_conditional.FFTMA_LS(domainsize=(500, 500), covmods=covmods)

	# OK for cv
	ok_cov = gs.Exponential(dim=2, var=1, len_scale=22, nugget=0.01)
	domainsize = (536, 536)
	mg = np.mgrid[[slice(0, domainsize[i], 1) for i in range(2)]].reshape(2,-1).T

	data_krig = gs.krige.Ordinary(ok_cov, [xy[:,0], xy[:,1]], cv, exact=True)
	z_data, s_data = data_krig([mg[:,0], mg[:,1]])
	z_data = z_data.reshape(domainsize)
	
	# FFTMA (without layer cop) for OK simulation for tau0fields
	fftma_ = fftma.FFTMA(domainsize=domainsize, covmod='0.01 Nug(0.0) + 0.99 Exp(22.0)')

	tau0fields = []
	for t in range(8):
		print('Kriging Simulation # {} '.format(t))
		rand_field = fftma_.simnew()
		cvrf = rand_field[xy[:,0], xy[:,1]]
		ok_rf = gs.krige.Ordinary(ok_cov, [xy[:,0], xy[:,1]], cvrf, exact=True)
		rf_data, ss = ok_rf([mg[:,0], mg[:,1]])
		rf_data = rf_data.reshape(domainsize)
		cfield = z_data + (rand_field - rf_data)
		tau0fields.append(cfield)
	tau0fields = np.array(tau0fields)


	# start conditional layer cop simulation
	Y = fftmals.condsim(xy, cv, tau0fields, nsteps=70, kbw=10)
	cond_fields.append(Y)

	# plot the results
	sim_cv = Y[xy[:, 0], xy[:, 1]]
	dif = np.sum((cv - sim_cv) ** 2)
	print(dif)

	als = np.array([-3,3])
	plt.figure()
	plt.scatter(cv ,sim_cv)
	plt.plot(als,als)
	plt.title('sq diff = {}'.format(dif))
	plt.savefig(r'scatter_{}.png'.format(s))
	plt.clf()
	plt.close()

	plt.figure()
	plt.imshow(Y, interpolation='nearest', origin='lower', cmap='jet', vmin=-3.6, vmax=3.6)
	plt.plot(xy[:,1], xy[:,0], 'x')
	plt.colorbar()
	plt.savefig(r'csimfield_{}.png'.format(s))
	plt.clf()
	plt.close()

	# save the conditional field
	np.save('csfield_{}.npy'.format(s), Y)

# plot mean and std
cond_fields = np.array(cond_fields)

plt.figure()
plt.imshow(np.mean(cond_fields, axis=0), interpolation='nearest', origin='lower', cmap='jet', vmin=-2.1, vmax=2.1)
plt.plot(xy[:,1], xy[:,0], 'x')
plt.colorbar()
plt.savefig('cond_mean.png', dpi=250)
plt.clf()
plt.close()

plt.figure()
plt.imshow(np.std(cond_fields, axis=0), interpolation='nearest', origin='lower', cmap='jet', vmin=0, vmax=1)
plt.plot(xy[:,1], xy[:,0], 'x')
plt.colorbar()
plt.savefig('cond_std.png', dpi=250)
plt.clf()
plt.close()

df = {}
df['l'] = []
df['val'] = []
for i in range(n_realisations):
	df['l'].append(list(np.arange(cv.shape[0])))
	df['val'].append(list(cond_fields[i][xy[:,0], xy[:,1]]))
df['l'] = np.concatenate(df['l'])
df['val'] = np.concatenate(df['val'])
df = pd.DataFrame(data=df, index=np.arange(n_realisations*cv.shape[0]))
df = df.sort_values(by='l').reset_index(drop=True)
dff = []
for i in range(cv.shape[0]):
	dff.append(df.loc[df.l == i].val)

fig, ax = plt.subplots(figsize=(24, 14))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(24)

ax.boxplot(dff)
ax.plot(np.arange(1,cv.shape[0]+1), cv, 'x', ms=18, mew=2)
plt.ylabel('values', fontsize=24)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'box_conditioning_values.png', dpi=250)
plt.clf()
plt.close()