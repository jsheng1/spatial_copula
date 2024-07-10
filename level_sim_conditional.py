#-------------------------------------------------------------------------------
# Name:        FFT-MA LayerSim
# Purpose:     Simulation of non-Gaussian spatial random fields
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     01/07/2022, Centre for Natural Gas, EAIT,
#                          The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------
import datetime
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
from helper_func import covariancefunction as covfun
from helper_func import varioFFT
try:
	import pyfftw
	fastfft = True
except:
	fastfft = False



class FFTMA_LS(object):
	def __init__(self,
				 domainsize = (100,100),
				 covmods    = ['0.01 Nug(0.0) + 0.99 Exp(2.)', '0.01 Nug(0.0) + 0.99 Exp(4.)'],
				 anisotropies = False, 	# requires tuple (scale 0, scale 1,...., scale n, rotate 0, rotate 1,..., rotate n-1)
				 						# note that scale is relative to range defined in covmod
				 reverse	= False,
				 periodic   = False,
				 fastfft 	= fastfft,
				 nthreads	= 4
				 ):

		self.anisotropies = anisotropies
		self.periodic = periodic
		self.covmods = covmods
		self.reverse = reverse
		self.fastfft = fastfft
		if self.fastfft: print('Using pyfftw.')
		self.nthreads = nthreads

		
		if len(domainsize) == 3:
			self.xyz = np.mgrid[[slice(0,n,1) for n in domainsize]].reshape(3,-1).T

		# adjust domainsize 
		cutoff = 32
		cutoffs = []
		for dim in domainsize:
			tsize = dim + cutoff
			# find closest multiple of 8 that is larger than tsize
			m8 = int(np.ceil(tsize/8.)*8.)
			cutoffs.append(m8 - dim)
		self.cutoff = np.array(cutoffs)

		self.domainsize = np.array(domainsize) + self.cutoff
		self.ndim       = len(self.domainsize)
		self.npoints    = np.prod(self.domainsize)

		self.grid = np.mgrid[[slice(0,n,1) for n in self.domainsize]]

		if self.anisotropies == False:
			# ensure periodicity of domain
			for i in range(self.ndim):
				self.domainsize = self.domainsize[:,np.newaxis]
			self.grid = np.min((self.grid, np.array(self.domainsize)-self.grid), axis=0)

			# compute distances from origin (--> wavenumbers in fourier space)
			h = ((self.grid**2).sum(axis=0))**0.5
			
			self.sqrtFFTQ = np.empty([len(self.covmods)] + list(h.shape))
			for ir, covm in enumerate(self.covmods):
				# covariances (in fourier space!!!)
				Q = covfun.Covariogram(h, covm)

				if self.fastfft:
					fft = pyfftw.builders.fftn(Q, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)
					FFTQ = np.abs(fft())
					# FFTQ = np.abs(pyfftw.interfaces.scipy_fftpack.fftn(Q, threads=self.nthreads))
					# FFTQ = np.abs(pyfftw.interfaces.numpy_fft.fftn(Q, threads=self.nthreads))
				else:
					FFTQ = np.abs(np.fft.fftn(Q))

				self.sqrtFFTQ[ir] = np.sqrt(FFTQ)
		else:
			self.apply_anisotropy()


	def simnewls(self):
		# normal random numbers
		u = np.random.standard_normal(size=self.sqrtFFTQ[0].shape)
		# fft of normal random numbers
		if self.fastfft:
			fft = pyfftw.builders.fftn(u, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)
			U = fft()
			# U = pyfftw.interfaces.scipy_fftpack.fftn(u, threads=self.nthreads)
			# U = pyfftw.interfaces.numpy_fft.fftn(u, threads=self.nthreads)
		else:
			U = np.fft.fftn(u)

		if self.reverse:
			tau = np.linspace(0, 1, self.sqrtFFTQ.shape[0] + 2)[1:-1][::-1]
		else:	
			tau = np.linspace(0, 1, self.sqrtFFTQ.shape[0] + 2)[1:-1]

		for ir, x in enumerate(tau):
			phi_tau = st.norm.ppf(x)
			# combine with covariance 
			GU = self.sqrtFFTQ[ir] * U
			# create field using inverse fft
			if self.fastfft:
				fft = pyfftw.builders.ifftn(GU, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)
				Y = np.real(fft())
				# Y = np.real(pyfftw.interfaces.scipy_fftpack.ifftn(GU, threads=self.nthreads))
				# Y = np.real(pyfftw.interfaces.numpy_fft.ifftn(GU, threads=self.nthreads))
			else:
				Y = np.real(np.fft.ifftn(GU))  

			if not self.periodic:
				# readjust domainsize to correct size (--> no boundary effects...)
				gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
														for i in range(self.ndim)]
				Y = Y[tuple(gridslice)]
				Y = Y.reshape(self.domainsize.squeeze()-self.cutoff)

			if self.reverse:
				if ir == 0:
					Y0 = np.copy(Y)
				else:
					Y0 = np.where(Y0 < phi_tau, Y, Y0)
			else:
				if ir == 0:
					Y0 = np.copy(Y)
				else:
					Y0 = np.where(Y0 > phi_tau, Y, Y0)

		return Y0

	def simnewls_given_RN(self, u):
		# fft of normal random numbers
		if self.fastfft:
			fft = pyfftw.builders.fftn(u, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)
			U = fft()
			# U = pyfftw.interfaces.scipy_fftpack.fftn(u, threads=self.nthreads)
			# U = pyfftw.interfaces.numpy_fft.fftn(u, threads=self.nthreads)
		else:
			U = np.fft.fftn(u)

		if self.reverse:
			tau = np.linspace(0, 1, self.sqrtFFTQ.shape[0] + 2)[1:-1][::-1]
		else:	
			tau = np.linspace(0, 1, self.sqrtFFTQ.shape[0] + 2)[1:-1]

		for ir, x in enumerate(tau):
			phi_tau = st.norm.ppf(x)
			# combine with covariance 
			GU = self.sqrtFFTQ[ir] * U
			# create field using inverse fft
			if self.fastfft:
				fft = pyfftw.builders.ifftn(GU, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)
				Y = np.real(fft())
				# Y = np.real(pyfftw.interfaces.scipy_fftpack.ifftn(GU, threads=self.nthreads))
				# Y = np.real(pyfftw.interfaces.numpy_fft.ifftn(GU, threads=self.nthreads))
			else:
				Y = np.real(np.fft.ifftn(GU)) 

			if not self.periodic:
				# readjust domainsize to correct size (--> no boundary effects...)
				gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
														for i in range(self.ndim)]
				Y = Y[tuple(gridslice)]
				Y = Y.reshape(self.domainsize.squeeze()-self.cutoff)

			if self.reverse:
				if ir == 0:
					Y0 = np.copy(Y)
				else:
					Y0 = np.where(Y0 < phi_tau, Y, Y0)
			else:
				if ir == 0:
					Y0 = np.copy(Y)
				else:
					Y0 = np.where(Y0 > phi_tau, Y, Y0)

		return Y0

	def apply_anisotropy(self):
		self.sqrtFFTQ = []
		for ir, cov in enumerate(self.covmods):
			ani = self.anisotropies[ir]
			# Create an array to stretch the distances
			stretchlist =[]
			for d in range(self.ndim):
				stretchdim = [0]*self.ndim
				stretchdim[d] = 1/ani[d]
				stretchlist.append(stretchdim)
			stretch = np.array(stretchlist)
			new_grid = self.grid.reshape(self.ndim, -1).T
			new_grid = np.dot(stretch, new_grid.T)
			new_grid = new_grid.reshape(self.grid.shape)

			# ensure periodicity of domain	
			for i in range(self.ndim):
				new_grid[i] = np.min((new_grid[i], np.max(new_grid[i]) + 1 - new_grid[i]), axis=0)

			# compute distances from origin (--> wavenumbers in fourier space)
			h = ((new_grid**2).sum(axis=0))**0.5

			# covariances (in fourier space!!!)
			Q = covfun.Covariogram(h, cov)

			# FFT of covariances and rotation
			nQ = np.fft.fftshift(Q)
			
			# I can't figure out how to make this more general...
			axeslist = []
			for d in range(self.ndim-1):
				axeslist.append((d, self.ndim-1))

			for d in range(self.ndim-1):
				angle = ani[self.ndim+d]
				nQ = scipy.ndimage.rotate(nQ, angle, axes=axeslist[d], reshape=False)
			nQ = np.fft.fftshift(nQ)

			if self.fastfft:
				fft = pyfftw.builders.fftn(nQ, overwrite_input=False, planner_effort='FFTW_ESTIMATE', threads=self.nthreads)
				FFTQ = np.abs(fft())
				# FFTQ = np.abs(pyfftw.interfaces.scipy_fftpack.fftn(nQ, threads=self.nthreads))
				# FFTQ = np.abs(pyfftw.interfaces.numpy_fft.fftn(nQ, threads=self.nthreads))
			else:
				FFTQ = np.abs(np.fft.fftn(nQ))

			self.sqrtFFTQ.append(np.sqrt(FFTQ))

		self.sqrtFFTQ = np.array(self.sqrtFFTQ)


	def invert_for_rn(self, field, sqrtFFTQ):
		u = np.real(np.fft.ifftn(np.fft.fftn(field)/sqrtFFTQ))
		return u


	def QQ_stdnorm(self, field):
		rankfield = (st.mstats.rankdata(field) - 0.5)/np.prod(field.shape)
		normfield = st.norm.ppf(rankfield)
		return normfield


	def condsim(self, cp, cv, taufields, nsteps=500, kbw=50):

		# get RN from first tau=0 (conditional) field
		uuold = self.invert_for_rn(taufields[0], self.sqrtFFTQ[(self.sqrtFFTQ.shape[0] - 1)//2])
		# simulate layer_sim_field with given RN
		Y2_init = self.simnewls_given_RN(uuold)
		# make it normal
		Y22 = self.QQ_stdnorm(Y2_init)
		# get values at cp
		new_cv_init = Y22[cp[:,0], cp[:,1]]

		# set up parameters 
		e1 = 2
		difold = np.sum((cv - new_cv_init)**e1)
		difref = np.sum((cv - new_cv_init) ** 2)
		print(difold, difref)
		
		# initial upper and lower bounds for section of circle
		tlb = -0.1
		tub = 0.1

		# initial window size for local optimization
		kloc_base = np.ones(cv.shape[0]).astype(int) * kbw
		kloc = np.copy(kloc_base)
		# mp is the maximum extention of the window size
		# which is calculated according to the obj func
		mp = 10

		# mask to select points that need improvement
		ix = np.where((cv - new_cv_init)**e1 > 0)

		# start with global optimization
		glob = True

		for i3 in range(nsteps):
			print('nsim:', i3)
		
			uunew = np.copy(uuold)
			
			if i3 < taufields.shape[0] - 1:
				# do global RM for RN
				uu = self.invert_for_rn(taufields[i3+1], self.sqrtFFTQ[(self.sqrtFFTQ.shape[0] - 1)//2])
				rns = np.concatenate((uuold[np.newaxis, :,:], uu[np.newaxis, :,:]))
				xsopt = self.circleopt(rns, cp[ix], cv[ix], e1, kloc[ix], glob=glob)
				print('xsopt:', xsopt)		
				uunew =  xsopt[0] * uuold + xsopt[1] * uu

			else:
				# local RM
				# draw new RN
				uu = np.random.standard_normal(size=taufields[0].shape)
				e1 = 2
				glob = False
				difold = difref

				if i3 == nsteps//2 + 10:
					# reduce section of circle
					tlb = -0.05
					tub = 0.05
					# reduce window size
					kloc_base = np.ones(cv.shape[0]).astype(int) * (kbw - 10)
					# change mp if required
					mp = 10
					q = (st.rankdata(diff_for_ix) - 0.5)/diff_for_ix.shape[0]
					q = (q * mp).astype(int)
					kloc = kloc_base + q
				
				# do local RM for RN
				for kk, xy in enumerate(cp[ix]):
					
					ilow = xy - kloc[ix][kk]
					ihigh = xy + kloc[ix][kk]

					uunew[ilow[0]:ihigh[0], ilow[1]:ihigh[1]] = uu[ilow[0]:ihigh[0], ilow[1]:ihigh[1]]
						
				rns = np.concatenate((uuold[np.newaxis, :,:], uunew[np.newaxis, :,:]))
				xsopt = self.circleopt(rns, cp[ix], cv[ix], e1, kloc[ix], tlb, tub, glob=glob)
		
				uunew = np.copy(uuold)
				for kk, xy in enumerate(cp[ix]):
					
					ilow = xy - kloc[ix][kk]
					ihigh = xy + kloc[ix][kk]

					uunew[ilow[0]:ihigh[0], ilow[1]:ihigh[1]] = xsopt[kk][1] * uu[ilow[0]:ihigh[0], ilow[1]:ihigh[1]] +\
																xsopt[kk][0] * uuold[ilow[0]:ihigh[0], ilow[1]:ihigh[1]]
						

			# calculate field with new rn		
			Y2 = self.simnewls_given_RN(uunew)
			Y2 = self.QQ_stdnorm(Y2)
			new_cv = Y2[cp[:, 0], cp[:, 1]]
			dif = np.sum((cv - new_cv) ** e1)

			if dif < difold:
				Yend = np.copy(Y2)
				difold = np.copy(dif) 
				uuold = np.copy(uunew)
				difref = np.sum((cv - new_cv) ** 2)
				print("Better", dif)
				print(np.corrcoef(cv, new_cv))

				if glob == False:
					diff_for_ix = (cv - new_cv)**e1
					ix = np.where(diff_for_ix > 1e-4)
					print(ix[0])
					q = (st.rankdata(diff_for_ix) - 0.5)/diff_for_ix.shape[0]
					q = (q * mp).astype(int)
					kloc = kloc_base + q

					# if all are < 1e-4 stop the loop
					if ix[0].shape[0] == 0:
						print('All done!')
						break

		return Yend

	def calc_field(self, weights, fields):
		return np.tensordot(weights, fields, axes=1)

	def get_points_on_circle(self, discr, usf):
		t = np.linspace(0, np.pi*2,(usf*discr)-(usf-1))
		return t

	def get_point_for_sinc(self, discr):
		self.t_s = np.linspace(-2*np.pi, np.pi*4, 3*discr-2)

	def get_samplepoints_on_circle(self, discr):
		t_s = np.linspace(0,np.pi*2,discr)
		xsample = np.array((np.cos(t_s),np.sin(t_s)))
		return xsample

	def get_samplepoints_close(self, discr, lb=-0.1, ub=0.1):
		t_s = np.linspace(lb, ub, discr)
		xsample = np.array((np.cos(t_s),np.sin(t_s)))
		return xsample

	def get_norm_rn_at_samplepoints(self, i, x, rns):
		rn = self.calc_field(x, rns)
		return rn
		

	def circleopt(self, rns, cp, cv, e1, kloc, tlb=-0.1, tub=0.1, glob=False):
		
		discr = 8 
		usf = 60

		if glob:
			xsample = self.get_samplepoints_on_circle(discr)
			self.get_point_for_sinc(discr)
			self.circlediscr = self.get_points_on_circle(discr, usf)

			# prepare sinc interpolation
			self.T = self.t_s[1] - self.t_s[0]		
			self.sincM = np.tile(self.circlediscr, (len(self.t_s), 1)) - np.tile(self.t_s[:, np.newaxis], (1, len(self.circlediscr)))
			self.sincMT = np.sinc(self.sincM/self.T)
		else:
			# prepare small section of circle
			xsample = self.get_samplepoints_close(discr - 1, tlb, tub)

		if glob:
			norm_rn = []
			for i,x in enumerate(xsample.T[:-1]):
				# calculate rn at samplepoints
				norm_rn.append(self.get_norm_rn_at_samplepoints(i, x, rns))
			norm_rn = np.array(norm_rn)
		else:
			norm_rn = []
			for i,x in enumerate(xsample.T):
				rnn = np.copy(rns[0])
				for kk, xy in enumerate(cp):		
					ilow = xy - kloc[kk]
					ihigh = xy + kloc[kk]					
					# calculate rn at samplepoints
					nrn = self.get_norm_rn_at_samplepoints(i, x, rns[:, ilow[0]:ihigh[0], ilow[1]:ihigh[1]])
					rnn[ilow[0]:ihigh[0], ilow[1]:ihigh[1]] = nrn
				norm_rn.append(rnn)
			norm_rn = np.array(norm_rn)


		# run level sim with all rn
		newFields = []
		for i in range(norm_rn.shape[0]):
			Y2 = self.simnewls_given_RN(norm_rn[i])
			Y2 = self.QQ_stdnorm(Y2)
			newFields.append(Y2)
		newFields = np.array(newFields)
		# get the values at the conditioning point locations
		self.nlvals = newFields[:, cp[:, 0], cp[:, 1]]

		if glob:
			# add the first one which is the same as the last (cyclic, i.e. same angle) 
			self.nlvals = np.vstack((self.nlvals, self.nlvals[0]))	
		
			# avoid the loop for sinc interp in matrix form
			intp_nlvals1 = np.concatenate((self.nlvals[:-1], self.nlvals, self.nlvals[1:])).T
			intp_nlvals = self.sinc_interp(intp_nlvals1)

			# find optimal solution
			objinter = np.sum((intp_nlvals - cv[:,np.newaxis])**e1, axis=0)
			objinter_min = np.min(objinter)
			ix = np.where(objinter == objinter_min)[0][0]
			xsopts = np.array((np.cos(self.circlediscr[ix]),np.sin(self.circlediscr[ix])))
		else:
			# polynomial interpolation
			t_s = np.linspace(tlb, tub, discr - 1)
			x = np.linspace(min(t_s), max(t_s), num=100)
			
			intp_nlvals = []
			for i in range(cv.shape[0]):				
				ifunc = scipy.interpolate.barycentric_interpolate(t_s, self.nlvals[:,i], x)
				intp_nlvals.append(ifunc)
			intp_nlvals = np.array(intp_nlvals)

			# get one angle for each location
			objinter = (intp_nlvals - cv[:,np.newaxis])**e1
			objinter_min = np.min(objinter, axis=1)

			# find optimal solutions from interpolated objective function
			xsopts = []
			for i in range(objinter.shape[0]):
				ix = np.where(objinter[i] == objinter_min[i])[0][0]
				xsopts.append(np.array((np.cos(x[ix]),np.sin(x[ix]))))
			xsopts = np.array(xsopts)

		return xsopts


	def sinc_interp(self, x):
		"""
		Interpolates x, sampled at "s" instants
		Output y is sampled at "u" instants ("u" for "upsampled")    
		"""
		y = np.dot(x, self.sincMT)
		return y





