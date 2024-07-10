#-------------------------------------------------------------------------------
# Name:        FFT-MA LayerSim
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
import scipy.interpolate as interpolate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from statsmodels.distributions.empirical_distribution import ECDF
from helper_func import fftma
import level_sim_conditional
import gstools as gs


# read BW groundwater nitrate data
nitrate = np.load(r'data/nitrate.npy')

# fit marginal using KDE
conc = nitrate[:,-1]
# optimize the kernelwidth
x = np.log(conc[conc > 0])
bandwidths = 10 ** np.linspace(-1.5, 0.5, 500)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
				{'bandwidth': bandwidths},
				cv=5)
grid.fit(x[:, None])

# use optimized kernel for kde
kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
kde.fit(x[:, None])

# build cdf and invcdf from pdf
xx = np.arange(x.min() - 4., x.max() + 0.2, 0.001)
logprob = np.exp(kde.score_samples(xx[:, None]))
cdf_ = np.cumsum(logprob) * 0.001
cdf_ = np.concatenate(([0.0], cdf_))
cdf_ = np.concatenate((cdf_, [1.0]))
xx = np.concatenate((xx, [x.max() + 0.2]))
xx = np.concatenate(([x.min() - 4.], xx))
cdf = interpolate.interp1d(xx, cdf_)
invcdf = interpolate.interp1d(cdf_, xx)

# standard normal values and coordinates
cv = st.norm.ppf(cdf(x))
xy = nitrate[:, :2].astype(int)


# SIMULATE
n_realisations = 10
cond_fields = []
for s in range(n_realisations):
	print('Simulate conditional realization # {}'.format(s))

	covmods = []
	nlev = 41 # has to be an odd number to get phi_(tau)=0
	tau = np.linspace(0.0, 1, nlev)

	r1 = 110 
	r2 = 100 
	n1 = 0.18
	n2 = 0.38
	for ii, r in enumerate(tau):
		nug = (1- tau[ii])*n1 + tau[ii]*n2
		pr = (1- tau[ii])*r1 + tau[ii]*r2
		covmod =  '{} Nug(0.0) + {} Exp({})'.format(nug, 1 - nug, pr)
		covmods.append(covmod)
	print(covmods)

	# initialize FFTMA_LS with the covmods defined above
	fftmals = level_sim_conditional.FFTMA_LS(domainsize=(1000, 1000), covmods=covmods)

	# OK for cv
	ok_cov = gs.Exponential(dim=2, var=1, len_scale=105, nugget=0.28)
	domainsize = (1032, 1032)
	mg = np.mgrid[[slice(0, domainsize[i], 1) for i in range(2)]].reshape(2,-1).T

	data_krig = gs.krige.Ordinary(ok_cov, [xy[:,0], xy[:,1]], cv, exact=True)
	z_data, s_data = data_krig([mg[:,0], mg[:,1]])
	z_data = z_data.reshape(domainsize)
	
	# FFTMA (without layer cop) for OK simulation for tau0fields
	fftma_ = fftma.FFTMA(domainsize=domainsize, covmod='0.28 Nug(0.0) + 0.72 Exp(105.0)')

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

	# return the field
	print('Start layer cop simulation')
	Y = fftmals.condsim(xy, cv, tau0fields, nsteps=100, kbw=2)

	# backtransform into value space
	sim_conc = st.norm.cdf(Y)
	sim_conc = invcdf(sim_conc)
	sim_conc = np.exp(sim_conc)

	np.save('concfield_{}.npy'.format(s), sim_conc)

	cond_fields.append(sim_conc)

	# plot the results
	sim_cv = Y[xy[:, 0], xy[:, 1]]
	dif = np.sum((cv - sim_cv) ** 2)
	print(dif)

	als = np.array([-3,3])
	plt.figure()
	plt.scatter(cv ,sim_cv)
	plt.plot(als,als)
	plt.title('sq diff = {}'.format(dif))
	plt.savefig(r'BW_gw_scatter_{}.png'.format(s))
	plt.clf()
	plt.close()

	plt.figure()
	plt.imshow(sim_conc, interpolation='nearest', origin='lower', cmap='jet')
	plt.colorbar()
	plt.savefig(r'conc_simfield_{}.png'.format(s))
	plt.clf()
	plt.close()

	

cond_fields = np.array(cond_fields)

plt.figure()
plt.imshow(np.mean(cond_fields, axis=0), interpolation='nearest', origin='lower', cmap='jet', vmin=-2.1, vmax=2.1)
plt.plot(xy[:,1], xy[:,0], 'x')
plt.colorbar()
plt.savefig('BW_gw_cond_mean.png', dpi=250)
plt.clf()
plt.close()

plt.figure()
plt.imshow(np.std(cond_fields, axis=0), interpolation='nearest', origin='lower', cmap='jet', vmin=0, vmax=1)
plt.plot(xy[:,1], xy[:,0], 'x')
plt.colorbar()
plt.savefig('BW_gw_cond_std.png', dpi=250)
plt.clf()
plt.close()

df = {}
df['l'] = []
df['val'] = []
for i in range(n_realisations):
	df['l'].append(list(np.arange(conc.shape[0])))
	df['val'].append(list(cond_fields[i][xy[:,0], xy[:,1]]))
df['l'] = np.concatenate(df['l'])
df['val'] = np.concatenate(df['val'])
df = pd.DataFrame(data=df, index=np.arange(n_realisations*conc.shape[0]))
df = df.sort_values(by='l').reset_index(drop=True)
dff = []
for i in range(conc.shape[0]):
	dff.append(df.loc[df.l == i].val)

fig, ax = plt.subplots(figsize=(24, 14))
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	label.set_fontsize(24)

ax.boxplot(dff)
ax.plot(np.arange(1,conc.shape[0]+1), conc, 'x', ms=18, mew=2)
plt.ylabel('values', fontsize=24)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(r'box_conditioning_values_BW_gw.png', dpi=250)
plt.clf()
plt.close()