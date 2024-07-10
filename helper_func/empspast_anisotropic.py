#-------------------------------------------------------------------------------
# Name:        Empirical spatial statistics
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     02.05.2018, UQ, Brisbane, QLD, Australia
# Copyright:   (c) Hoerning 2018
#-------------------------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.spatial as sp
import datetime
import IPython
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



class progressbar(object):
    __ladebalken = np.linspace(1,10,10)
    def __init__(self, niteritems, talk_to_me=True):
        """
        niteritems ... number of times the loop is crossed
        """
        self.n = niteritems
        self.talk_to_me = talk_to_me
    def __call__(self,i):
        if self.talk_to_me == True:
            if int((i/float(self.n))*10) in self.__ladebalken:
                if self.__ladebalken[0] == 5:
                    print( '50%',)
                else:
                    print( '.',)
                self.__ladebalken = np.delete(self.__ladebalken, 0)
            elif i == 0: print( '0%',)
            elif i == self.n-1: print( '100%')
##-----------------------------------------------------------------------------#
class empspast_isotropic_unstructured(object):
    def __init__(
            self,
            xyz,        # [x1, x2, ..., xd] n coordinates in d-dimensional space
            values,     # [p1, p2, ..., pk] k value vectors of length n

            coordinatesystem = 'cartesian', # 'cartesian' or 'spherical'
            talk_to_me = True,
                    ):

        self.xyz    = np.array(xyz)
        self.values = np.array(values)
        self.coordinatesystem = coordinatesystem

        # number of points
        self.npoints = self.xyz.shape[0]
        if talk_to_me==True: print( 'Number of points:', self.npoints)

        # bring to correct shape
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)

        # ranked values
        self.values_ranked = (
                st.mstats.rankdata(self.values,axis=0) - 0.5) / self.npoints

        self.talk_to_me             = talk_to_me
        if self.talk_to_me == True:
            print( '---------------------------------')
            print( '- EMPIRICAL SPATIAL STATISTICS  -')
            print( '---------------------------------')

        # make coordinate array two-dimensonal
        if self.xyz.ndim == 1:
            self.xyz = self.xyz.reshape(len(self.xyz), 1)

        # number of (spatial) dimensions
        self.ndim = self.xyz.shape[1]
        if talk_to_me==True: print( 'Number of spatial dimensions:', self.ndim)

        # number of datasets
        self.ndatasets = self.values.shape[1]
        if self.talk_to_me==True: print( 'Number of datasets: %i'%self.ndatasets)
    ##------------------------------------------------------------------------##
##    @profile
    def calc_stats( self,
                    lagbounds = None,    # boundaries of lag classes
                    nbins = 25,          # number of bins for copula
                    ang_bounds_maj = [20, 30],
                    ang_bounds_min = [110, 120] 
                   ):

        # LAGSBOUNDS & cutoff
        # embed()
        if lagbounds is None:
            dbig = (((np.max(self.xyz, axis=0)
                                - np.min(self.xyz, axis=0))**2).sum())**0.5
                        # this is >= biggest distance in dataset
            self.cutoff    = dbig / 3.0
            self.lagbounds = np.linspace(0,self.cutoff,11)
        else:
            self.cutoff    = lagbounds[-1]
            self.lagbounds = np.array(lagbounds)
        self.nlags = len(self.lagbounds)-1

        starttime = datetime.datetime.now() # to measure time of calculation

        self.angle_bounds_pos_maj = np.array(ang_bounds_maj)
        self.angle_bounds_neg_maj = -180 + self.angle_bounds_pos_maj
        self.angle_bounds_pos_min = np.array(ang_bounds_min)
        self.angle_bounds_neg_min = -180 + self.angle_bounds_pos_min

        if self.talk_to_me==True:
            print( 'Number of bins for empirical copula:', nbins)

        # PREPARE lists of dictionaries for output statistics
        # (one dictionary for every dataset)
        self.statlist_uni = []
        self.statlist_biv = []
        for dset in range(self.ndatasets):
            adict = {}

            # bivariate
            adict['h'] = np.zeros(self.nlags)
            adict['h_major'] = np.zeros(self.nlags)
            adict['h_minor'] = np.zeros(self.nlags)
            adict['npairs'] = np.zeros(self.nlags)
            adict['npairs_major'] = np.zeros(self.nlags)
            adict['npairs_minor'] = np.zeros(self.nlags)

            adict['bivariate_copula'] = np.zeros((self.nlags,nbins,nbins))+0.0000001

            adict['variogram'] = np.zeros(self.nlags)
            adict['variogram_major'] = np.zeros(self.nlags)
            adict['variogram_minor'] = np.zeros(self.nlags)
            adict['covariance'] = np.zeros(self.nlags)
            adict['correlation'] = np.zeros(self.nlags)
            adict['R'] = np.zeros(self.nlags) # rank correl
            adict['A'] = np.zeros(self.nlags) # asymmetry (Jing)
            adict['A_t'] = np.zeros(self.nlags) # Asymmetry new
            adict['A_t_normed'] = np.zeros(self.nlags) # Asymmetry new normed
            adict['K_t'] = np.zeros(self.nlags) # biv. kurtosis

            self.statlist_uni.append({})
            self.statlist_biv.append(adict)

        # CALCULATE UNIVARIATE STATISTICS
        for dset in range(self.ndatasets):
            self.statlist_uni[dset]['mean']   = np.mean(self.values[:,dset])
            self.statlist_uni[dset]['median'] = np.median(self.values[:,dset])
            self.statlist_uni[dset]['var']    = np.var(self.values[:,dset])**0.5
            self.statlist_uni[dset]['skew']   = st.skew(self.values[:,dset])
            self.statlist_uni[dset]['kurt']   = st.kurtosis(self.values[:,dset])

        # LOOP OVER LOCATIONS
        # ----------------------------------------------------------------------
        if self.talk_to_me: print( 'calculate empirical bivariate copulas:')
        ladebalken = progressbar(self.npoints,self.talk_to_me)
        for ii in range(self.npoints):      # ii is the current point
            ladebalken(ii) # LADEVORGANG

            # Find possible points for pairs
            # save possible pair-ids in ix
            # ------------------------------------------------------------------
            if self.coordinatesystem == 'spherical':
                # 1) calc distances
                '''
                Phi = Breitengrad:      -90 <=Phi <=90
                Lambda = Laengengrad:   0<= Lamda <= 360
                but: also negative values are possible if consistent!
                Calculates a orthodrome-distance matrix of points on a sphere
                '''
                Lambda = self.xyz[:,0]
                Phi    = self.xyz[:,1]

                # Ins Bogenmass
                LB    = Lambda  *2 *np.pi /360
                PB    = Phi     *2 *np.pi /360
                LA = LB[ii]
                PA = PB[ii]
                # LA and PA are Lambda an Phi of current position
                # LB and PB are Lambda an Phi of all positions

                # calculate orthodromes
                # (the distances between A and B on the surface of a sphere)
                psi = np.arccos(np.sin(PA)*np.sin(PB)+np.cos(PA)*np.cos(PB)*np.cos(LB-LA))
                radius=6370 # of the earth
                d = psi * radius

                ix = np.where((d<self.cutoff)&(d!=0))[0]  # indices of possible partners

            elif self.coordinatesystem == 'cartesian':
                # calc distances
                # all distances from current point
                d = (((self.xyz[ii][np.newaxis] - self.xyz)**2).sum(axis=1))**0.5
                # isotrop indices of possible partners for a pair:
                ix = np.where((d<=self.cutoff)&(d!=0))[0]  # indices of possible partners
                # calculate angles
                angles = np.rad2deg(np.arctan2(self.xyz[ii][0] - self.xyz[ix][:,0], self.xyz[ii][1] - self.xyz[ix][:,1]))
                
                aix_major = np.where(((angles >= self.angle_bounds_pos_maj[0]) & (angles <= self.angle_bounds_pos_maj[1]))
                            | ((angles >= self.angle_bounds_neg_maj[0]) & (angles <= self.angle_bounds_neg_maj[1])))

                aix_minor = np.where(((angles >= self.angle_bounds_pos_min[0]) & (angles <= self.angle_bounds_pos_min[1]))
                            | ((angles >= self.angle_bounds_neg_min[0]) & (angles <= self.angle_bounds_neg_min[1])))
                # update ix
                ix_maj = ix[aix_major]
                ix_min = ix[aix_minor]

            # now, ix are the indices of all possible partners
            # with maximum distance of cutoff

            # LOOP OVER LAGS
            for jj in range(self.nlags):
                lb = self.lagbounds[jj]
                ub = self.lagbounds[jj+1]
                jx = ix[np.where((d[ix]>lb)&(d[ix]<=ub))]
                            # the points in right distance from current point
                jx_maj = ix_maj[np.where((d[ix_maj]>lb)&(d[ix_maj]<=ub))]
                jx_min = ix_min[np.where((d[ix_min]>lb)&(d[ix_min]<=ub))]

                for ixnum, ixx in enumerate([jx, jx_maj, jx_min]): 
                    
                    if len(ixx) > 0: # if there are points in that distance

                        # LOOP OVER DATASETS
                        for dset in range(self.ndatasets):

                            if ixnum == 0:
                                # CALCULATE STATS
                                # ---------------
                                # distances summed up
                                # (divide by number of pairs of points later)
                                self.statlist_biv[dset]['h'][jj] += d[ixx].sum()
                                # number of pairs of points
                                self.statlist_biv[dset]['npairs'][jj] += len(ixx)

                                x1 = self.values[:,dset][ii]
                                x2 = self.values[:,dset][ixx]
                                u1 = self.values_ranked[:,dset][ii]
                                u2 = self.values_ranked[:,dset][ixx]

                                # bivariate stats
                                # (divide by number of pairs of points later)
                                # variogram
                                self.statlist_biv[dset]['variogram'][jj] += ((x1-x2)**2).sum()
                                # covariance
                                m = self.statlist_uni[dset]['mean']
                                c = ((x1-m)*(x2-m)).sum()
                                self.statlist_biv[dset]['covariance'][jj] += c
                                # correlation
                                var = self.statlist_uni[dset]['var']
                                self.statlist_biv[dset]['correlation'][jj] += c/var

                                # empirical bivariate PDF :
        ##                        self.statlist_biv[dset]['bivariate_copula'][jj] += np.histogramdd(
        ##                                        np.array((u1*np.ones(u2.shape),u2)).T,
        ##                                        bins=nbins,
        ##                                        range=[[0,1],[0,1]],
        ##                                        normed=False
        ##                                        )[0]   # norm later

                                ui = np.vstack((u1*np.ones(u2.shape),u2))
                                ui = np.floor(ui*nbins)
                                weightsi = np.ones(u2.size)
                                grid = scipy.sparse.coo_matrix((weightsi, ui),
                                                        shape=(nbins, nbins)).toarray()
                                self.statlist_biv[dset]['bivariate_copula'][jj] += grid
                            
                            elif ixnum == 1:
                                # CALCULATE STATS
                                # ---------------
                                # distances summed up
                                # (divide by number of pairs of points later)
                                self.statlist_biv[dset]['h_major'][jj] += d[ixx].sum()
                                # number of pairs of points
                                self.statlist_biv[dset]['npairs_major'][jj] += len(ixx)

                                x1 = self.values[:,dset][ii]
                                x2 = self.values[:,dset][ixx]


                                # bivariate stats
                                # (divide by number of pairs of points later)
                                # variogram
                                self.statlist_biv[dset]['variogram_major'][jj] += ((x1-x2)**2).sum()

                            elif ixnum == 2:
                                # CALCULATE STATS
                                # ---------------
                                # distances summed up
                                # (divide by number of pairs of points later)
                                self.statlist_biv[dset]['h_minor'][jj] += d[ixx].sum()
                                # number of pairs of points
                                self.statlist_biv[dset]['npairs_minor'][jj] += len(ixx)

                                x1 = self.values[:,dset][ii]
                                x2 = self.values[:,dset][ixx]


                                # bivariate stats
                                # (divide by number of pairs of points later)
                                # variogram
                                self.statlist_biv[dset]['variogram_minor'][jj] += ((x1-x2)**2).sum()


        # delete distance classes, where less than nmin tuples were found
        if self.talk_to_me: print( 'correct empirical copulas and calc stats...')
        nmin_pairs = 10
        for dset in range(self.ndatasets):
            # 2d
            ix = np.where(self.statlist_biv[dset]['npairs']<nmin_pairs)[0]

            for key in self.statlist_biv[dset].keys():
                self.statlist_biv[dset][key] = np.delete(
                                        self.statlist_biv[dset][key], ix, axis=0)

        for dset in range(self.ndatasets):   # LOOP OVER DATASETS
            # norm everything
            # ---------------
            # biv. stats
            for key in self.statlist_biv[dset].keys():
                if not ((key=='npairs')|(key=='bivariate_copula')|
                        (key=='npairs_major')|(key=='npairs_minor')|
                        (key=='variogram_minor')|(key=='variogram_major')|
                        (key=='h_minor')|(key=='h_major')):
                    self.statlist_biv[dset][key] /= self.statlist_biv[dset]['npairs']
            self.statlist_biv[dset]['variogram'] /= 2 # just for the variogram
            self.statlist_biv[dset]['variogram_major'] /= self.statlist_biv[dset]['npairs_major']
            self.statlist_biv[dset]['h_major'] /= self.statlist_biv[dset]['npairs_major']
            self.statlist_biv[dset]['variogram_major'] /= 2
            self.statlist_biv[dset]['variogram_minor'] /= self.statlist_biv[dset]['npairs_minor']
            self.statlist_biv[dset]['h_minor'] /= self.statlist_biv[dset]['npairs_minor']
            self.statlist_biv[dset]['variogram_minor'] /= 2
            

            # bivar. copulas
            cs = self.statlist_biv[dset]['bivariate_copula'].mean(
                                 axis=-1).mean(axis=-1)[:,np.newaxis,np.newaxis]
            self.statlist_biv[dset]['bivariate_copula'] /= cs


            # now correct copulas & calc. rank stats
            # grid of emp. 2d-copula
            us2 = (np.mgrid[0:nbins,0:nbins] + 0.5) / nbins
            us2 = us2#[:,::-1]
            # loop over lags that had data values
            for jj in range(self.statlist_biv[dset]['h'].shape[0]):
                cc2 = self.statlist_biv[dset]['bivariate_copula'][jj]

                # correct empirical 2d copula
                krnlwidth = np.max((nbins/25.,1.0))
                cc2 = correct_emp_ndim_cop(cc2, krnlwidth=krnlwidth, niter=10)
                self.statlist_biv[dset]['bivariate_copula'][jj] = cc2

            # calculate rank statsistics from corrected empirical copulas
            # -----------------------------------------------------------
            R,A,At,An,Kt = RAAtAnKt_from_biv_copula(
                                    self.statlist_biv[dset]['bivariate_copula'])

            self.statlist_biv[dset]['R'] = R            # rank correlation
            self.statlist_biv[dset]['A'] = A            # asymmertry classical
            self.statlist_biv[dset]['A_t'] = At         # asymmertry new
            self.statlist_biv[dset]['K_t'] = Kt         # biv. kutosis new
            Amax = Amax_from_R(R)                       # normed asymmetry
            self.statlist_biv[dset]['A_t_normed'] = At / Amax # normed At

        if self.talk_to_me == True:
            print( 'computation time:', str(datetime.datetime.now()-starttime))
            print( '---------------------------------')
    ##-------------------------------------------------------------------------#
    def average_statlists(self):
        # Calculate the mean
        statlist_uni_mean = {}
        for key in self.statlist_uni[0]:
            for dset in range(self.ndatasets):
                if dset == 0:
                    statlist_uni_mean[key] = self.statlist_uni[dset][key]
                else:
                    statlist_uni_mean[key] += self.statlist_uni[dset][key]
                statlist_uni_mean[key] /= self.ndatasets
        statlist_biv_mean = {}
        for key in self.statlist_biv[0]:
            for dset in range(self.ndatasets):
                if dset == 0:
                    statlist_biv_mean[key] = self.statlist_biv[dset][key]
                else:
                    statlist_biv_mean[key] += self.statlist_biv[dset][key]
                statlist_biv_mean[key] /= self.ndatasets
        return statlist_uni_mean, statlist_biv_mean
    ##-------------------------------------------------------------------------#

    def save_stats(self, path_n_prefix='out', dset=0):
        biout = '#h v c r a\n'
        for i in range(self.statlist_biv[dset]['h'].shape[0]):
            biout += '%f %f %f %f %f\n'%(
                                        self.statlist_biv[dset]['h'][i],
                                        self.statlist_biv[dset]['variogram'][i],
                                        self.statlist_biv[dset]['covariance'][i],
                                        self.statlist_biv[dset]['R'][i],
                                        self.statlist_biv[dset]['A'][i],
                                        )
        fobj = open(path_n_prefix+'.bivstat.empspast', 'w')
        fobj.write(biout)
        fobj.close()

        fobj = open(path_n_prefix+'.bivcops.empspast', 'w')
        fobj.write('# bivariate copulas\n')
        for i in range(self.statlist_biv[dset]['h'].shape[0]):
            fobj.write('# h=%f\n'%self.statlist_biv[dset]['h'][i])
            np.savetxt(fobj, self.statlist_biv[dset]['bivariate_copula'][i])
        fobj.close()

    def plt_stats(  self,
                    interpolation='nearest'
                    ):
        if self.talk_to_me==True:
            print( '- PLOT spatial statistics       -')
        fontsize = 10
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['legend.fontsize'] = fontsize
        plt.rcParams['figure.subplot.bottom'] = 0.05
        plt.rcParams['figure.subplot.top'] = 0.95
        plt.rcParams['figure.subplot.left'] = 0.07
        plt.rcParams['figure.subplot.right'] = 0.94
        plt.rcParams['figure.subplot.wspace'] = 0.2

        # average stats:
        statlist_uni_mean, statlist_biv_mean = self.average_statlists()


        # BEGIN FIGURE
        # ------------
        fig = plt.figure(figsize=(10,10),num=667)
        fig.clf()

        # FIELD ---------------------------------------------------------------#
        if self.ndim == 3:  # 3D
            axfield = plt.subplot2grid((4,2), (0,0), projection='3d', rowspan=2)
            axfield.scatter(self.xyz[:,0],self.xyz[:,1],self.xyz[:,2], c=self.values_ranked[:,0], marker='o',linewidth=0)

        elif self.ndim == 2:  # 2D
            try:   # gridded data???
                axfield = plt.subplot2grid((4,2), (0,0),rowspan=2,aspect='equal')
                nx = np.array(list(set(self.xyz[:,0]))).shape[0]
                ny = np.array(list(set(self.xyz[:,1]))).shape[0]
                for i in [0,1]: # sort:
                    ix = np.argsort(self.xyz[:,i], kind='mergesort')
                    self.xyz                = self.xyz[ix]
                    self.values         = self.values[ix]
                    self.values_ranked  = self.values_ranked[ix]

                zi = self.values_ranked[:,0].reshape(ny,nx)

                plt.imshow( zi,
                            extent=(    self.xyz[:,0].min(),
                                        self.xyz[:,0].max(),
                                        self.xyz[:,1].min(),
                                        self.xyz[:,1].max(),),
                            interpolation='nearest',
                            origin='lower',
                            ##cmap=plt.cm.afmhot
                            )
                plt.xlabel('x')
                plt.ylabel('y')
                cb = plt.colorbar(shrink=0.6)
                for t in cb.ax.get_yticklabels():
                     t.set_fontsize(fontsize)
##                plt.plot(self.xyz[:,0],self.xyz[:,1], ',', color='grey',alpha=0.3)
##                plt.xlim(xi.min()-dx/2.0,xi.max()+dx/2.0)
##                plt.ylim(yi.min()-dy/2.0,yi.max()+dy/2.0)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

            except: # plot bubbles...
                fig.clf()
                axfield = plt.subplot2grid((4,2), (0,0),rowspan=2,aspect='equal')
                ind = np.arange(self.npoints)
                np.random.shuffle(ind)
                plt.scatter(    self.xyz[ind,0],
                                self.xyz[ind,1],
                                c=self.values_ranked[ind,0],
                                marker='o',linewidth=0)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

        elif self.ndim == 1:  # 1D
            axfield = plt.subplot2grid((4,2), (0,0),rowspan=2)
            plt.plot(self.xyz[:,0],self.values_ranked[:,0],',')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

        # HISTOGRAM -----------------------------------------------------------#
        plt.subplot2grid((4,2), (2,0))
        statstring = 'mean = %f\nvar = %f\nskew = %f\nkurt = %f'%(
            statlist_uni_mean['mean'],
            statlist_uni_mean['var'],
            statlist_uni_mean['skew'],
            statlist_uni_mean['kurt'], )
        plt.hist(self.values.flatten(), bins=25, density=True, color='blue', label=statstring)
        leg = plt.legend(loc=2, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # functions of distances ----------------------------------------------#

        # CORRELATION FUNCTIONS
        h = statlist_biv_mean['h']
        R = statlist_biv_mean['R']
        v = statlist_biv_mean['variogram']
        C = statlist_biv_mean['covariance']

        # rank correlation function
        ax0 = plt.subplot2grid((4,2), (0,1))
        plt.plot(h,R, '1-', color='blue',label='$R(h)$',linewidth=1.5)
        plt.plot([0,h.max()], [0,0], '--', color='blue',linewidth=0.5,alpha=0.6)
        leg = plt.legend(loc=5, fancybox=True)
        leg.get_frame().set_alpha(0.8)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.yticks([0.0,0.5,1.0])
        plt.ylim(-0.2,1.0)
        plt.ylabel('Rank Correlation', fontsize=fontsize)

        # variogram & covariogram
        ax2 = plt.twinx()
        plt.plot(h,v, 'x-', color='black', label='$\gamma(h)$',alpha=0.7)
        plt.plot(h,C, '.-', color='black', label='$C(h)$',alpha=0.7)
        plt.plot([0,h.max()], [0,0], '--', color='black',linewidth=0.5,alpha=0.4)
        leg = plt.legend(loc=1, fancybox=True)
        leg.get_frame().set_alpha(0.8)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlim(0,)
        plt.ylabel('Variogram / Covariance', fontsize=fontsize)


        # ASYMMETRIES
        A = statlist_biv_mean['A']
        Atn = statlist_biv_mean['A_t_normed']

        plt.subplot2grid((4,2), (1,1), sharex=ax0)
        # A_t normed by Amax
        plt.plot(h,Atn, '1-', color='blue',label='$A_{normed}(h)$',linewidth=1.5)
        plt.plot([0,h.max()], [0,0], '--', color='blue',linewidth=0.5,alpha=0.6)
        plt.ylabel('normed Asymmetry', fontsize=fontsize)
        plt.ylim(-0.35,0.35)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        leg = plt.legend(loc=1, fancybox=True)
        leg.get_frame().set_alpha(0.8)
        # A
        ax2 = plt.twinx()
        plt.plot(h,A, '.-', color='black',label='$A_{t}(h)$',alpha=0.7)
        plt.ylabel('$A_{t}$', fontsize=fontsize)
        plt.yticks([-0.01,0,0.01])
        plt.ylim(-0.012,0.012)
        plt.yticks(fontsize=fontsize)
        leg = plt.legend(loc=4, fancybox=True)
        leg.get_frame().set_alpha(0.8)


        # 2d-kurtosis
        K = statlist_biv_mean['K_t']
        plt.subplot2grid((4,2), (2,1), sharex=ax0)
        plt.plot(h,K,'+-',color='blue',  label='$K$')
        leg = plt.legend(loc=1, fancybox=True)
        leg.get_frame().set_alpha(0.6)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('biv. Kurtosis', fontsize=fontsize)
        plt.xlabel('distance')


        # BIVARIATE COPULA DENSITIES ------------------------------------------#
        c = statlist_biv_mean['bivariate_copula']
        cmax = 3
        grid = ImageGrid(   fig,
                            414, # similar to subplot(111)
                            nrows_ncols = (1,c.shape[0]), # creates grid of axes
                            #grids = c.shape[0],    # number of grids
                            axes_pad=0.05, # pad between axes in inch.
                            share_all=True,
                            label_mode = '1',
                            )
        for i in range(c.shape[0]):
            im = grid[i].imshow( c[i],
                                 extent=(0.,1.,0.,1.),
                                 origin='lower',
                                 interpolation=interpolation,
                                 vmin=0,
                                 vmax=cmax
                                 )
        grid[0].set_xticks([0,0.5,1])
        grid[0].set_yticks([0,0.5,1])

        axins = inset_axes(grid[-1],
                   width="8%", # width = 10% of parent_bbox width
                   height="100%", # height : 50%
                   loc=3,
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=grid[-1].transAxes,
                   borderpad=0,
                   )
        plt.colorbar(im, cax=axins, ticks=[1,2,3])

        #----------------------------------------------------------------------#
        #plt.rcdefaults()    # restore default rc parameters
        #plt.tight_layout()
        return fig
##-----------------------------------------------------------------------------#
def correct_emp_ndim_cop(cc, krnlwidth=1, niter=10):
    cc = np.array(cc)
    nbins = cc.shape[0]
    ndim  = cc.ndim

    cc = cc+0.0000000000001   # zeros suck

    # smoothing...
    if krnlwidth >= 1.0:
        dx = np.ceil(krnlwidth).astype(int)
        xyz = np.mgrid[[slice(-dx,dx+1,1.) for i in range(ndim)]]
        g = np.exp(-(xyz**2./float(krnlwidth)).sum(axis=0))
        g = g/g.sum()
        cc = scipy.signal.convolve(cc,g,mode='same')# 'valid', 'same', 'full'

    # correct to have uniform marginal
    for i in range(niter):# how often???
        cc = cc/cc.sum()*nbins**ndim   # good total mass!?!

        mmax = 0
        mmin = 2

        cmprod  = np.ones(cc.shape) # this is goig to be the product of the marginals
        for dd in range(ndim):
            # margin in current dimension
            c0 = cc.swapaxes(0,dd)
            for mm in range(ndim-1):
                c0 = c0.sum(axis=0) / nbins
            # avoid overcompensation per iteration
            c0 = (c0+1.)/2.
            # bring to right shape
            c0 = np.ones(cc.shape) * c0
            ##cmprod *= c0.swapaxes(-1,dd)
            cmprod *= c0.swapaxes(0,dd)

            mmax = max(mmax, c0.max())
            mmin = min(mmin, c0.min())

        # correction
        cc = cc / cmprod

        # check if converged
        if mmax < 1.01:
            if mmin > 0.99:
                break
    return cc
##-----------------------------------------------------------------------------#
def RAAtAnKt_from_biv_copula(c):
    """
    c is a bivariate copula discretized in n x n pixels
    c can also be 3-dimensional,
        then the first dimension is the number of lags
        and the results have this length
    """
    cs = np.array(c)
    if cs.ndim==2:
        cs = cs[np.newaxis,:,:]
    nbins = cs.shape[-1]
    ndim = cs.ndim - 2  # the dimension of the biv. copula function
    nlags = cs.shape[:-2]

    # indices of domain grid
    gridslice = [slice(0,nlags[i],1) for i in range(ndim)]
    grid = (np.array(np.mgrid[gridslice]))
    lagix = grid.reshape(ndim,-1).T

    # make grid of copula
    u = (np.mgrid[0:nbins,0:nbins] + 0.5) / nbins

    # make empty output arrays
    R   = np.zeros(nlags)
    A   = np.zeros(nlags)
    At  = np.zeros(nlags)
    An  = np.zeros(nlags)
    Kt  = np.zeros(nlags)

    for ix in lagix:
        i = tuple(ix)
        c = cs[i]
        w = c/c.sum()
        # rank correlation
        R[i]  = 12*np.sum(np.prod(u-0.5, axis=0)*w)
        # asymmetry
        A[i] = np.sum(((u[0]-0.5)**2*(u[1]-0.5)+(u[0]-0.5)*(u[1]-0.5)**2)*w)
        # asymmertry new
        At[i] = np.sum(((np.sum(u, axis=0)-1.0 )**3)*w)
        # normed asymmetry
        Amax = Amax_from_R(R[i])
        An[i] = At[i] / Amax
        # biv. kurtosis
        Kt[i] = np.sum(((np.sum(u, axis=0)-1.0 )**2)*w)

    return R,A,At,An,Kt
##-----------------------------------------------------------------------------#
def spearman_rho_from_r(r):
    return 2*np.sin(r*np.pi/6)
##-----------------------------------------------------------------------------#
def spearman_r_from_rho(rho):
    return 6/np.pi*np.arcsin(rho/2.0)
##-----------------------------------------------------------------------------#
def Amax_from_R(R):
    return (1-((1-R)/2)**(1.0/3.0))*(1-R)/2.0
##-----------------------------------------------------------------------------#



