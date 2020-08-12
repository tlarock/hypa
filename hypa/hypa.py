import numpy as np
import scipy.sparse as sp
import pathpy as pp
from scipy.stats import hypergeom
from .computexi import computeXiHigherOrder, fitXi
from julia.Distributions import Hypergeometric, cdf, logcdf

class Hypa:
    '''
    Class for computing hypa scores on a DeBruijn graph given pathway data.
    '''
    def __init__(self, paths, ghype_r=None):
        """
        Initialize class with pathpy.paths object.

        parameters
        -----------

        paths: Paths
            Paths object containing the pathway data.
        """

        self.paths = paths


    def initialize_xi(self, k=2, sparsexi=True, redistribute=True, xifittol=1e-2, constant_xi=False, verbose=True):
        r"""
        Initialize the xi matrix for the paths object.

        Parameters
        ----------
        k: int
            Order to compute xi at
        sparsexi: logical
            If True, use scipy sparse matrices for computations. Default True.
        redistribute: logical
            If True, call fitXi on the matrix to redistribute excess weights. Default True.
        xifittol: float
            Error tolerance in expected weight for fitXi call. Ignored if redistribute is False.
        constant_xi: logical
            If True, also compute the Xi matrix that represents the null model where all weight is equally distributed.
        verbose: logical
            If True, print out details of what is happening.


        """
        if verbose:
            print('Computing the k={} order Xi...'.format(k))

        ## Assign k
        self.k = k
        ## TODO assuming sparse matrix here
        ## Compute Xi. Also returns a network object.
        self.Xi, self.hypa_net = computeXiHigherOrder(self.paths, k=self.k, sparsexi=sparsexi, constant_xi=False)
        self.adjacency = self.hypa_net.adjacency_matrix()

        if redistribute:
            if verbose:
                print('Fitting Xi...')

            ## TODO again assuming sparse matrix
            self.Xi = fitXi(self.adjacency, self.Xi, sparsexi=sparsexi, tol=xifittol, verbose=verbose)

        if constant_xi:
            self.Xi_cnst, _ = computeXiHigherOrder(self.paths, k=self.k, sparsexi=sparsexi, constant_xi=constant_xi)

        self.adjacency = self.hypa_net.adjacency_matrix()


    def construct_hypa_network(self, k=2, log=True, sparsexi=True, redistribute=True, xifittol=1e-2, baseline=False, constant_xi=False, verbose=True):
        """
        Function to compute the significant pathways from a Paths object.

        parameters
        -----------

        paths: Paths
            Paths object containing the pathway data.
        order: int
            The order at which the significance should be computed.
        pthresh: floatd
            Significance threshold for over/under-represented transitions.
        log: logical
            If True, compute and return pvals as log(p)


        Return
        -----------

        hypa_net: pathpy.Network
            pathpy network representing transitions at order _k_. Edge data
                includes the observed frequency, hypa score, and xi value.
        network: pathpy.Network
            pathpy network corresponding to the higher order Xi matrix
        Xi: numpy.ndarray
            The Xi matrix computed based on the path data


        """
        def add_edge(u, v, xival, xisum, adjsum, reverse_name_dict):
            #import rpy2.robjects as ro
            #import rpy2.robjects.numpy2ri
            #from rpy2.robjects.packages import importr
            source, target = reverse_name_dict[u],reverse_name_dict[v]
            pval = self.compute_hypa(self.adjacency[u,v], xival, xisum, adjsum, log_p=True)
            if xival > 0:
                try:
                    ## What if I return (source, target, attr) and create the dictionary after?
                    self.hypa_net.edges[(source, target)]['pval'] = pval
                except Exception as e:
                    attr = {'weight': 0.0, 'pval':pval, 'xi':xival}
                    self.hypa_net.add_edge(source, target, **attr)
            return 1


        ## assign k
        self.k = k

        ## create network and Xi matrix
        if not baseline:
            self.initialize_xi(k=self.k, sparsexi=sparsexi, redistribute=redistribute, xifittol=xifittol, constant_xi=constant_xi, verbose=verbose)

            if not constant_xi:
                xi = self.Xi
            else:
                xi = self.Xi_cnst

            xisum = xi.sum()
            xicoo = sp.coo_matrix(xi)
        else:
            xicoo = sp.coo_matrix(self.adjacency)

        reverse_name_dict = {val:key for key,val in self.hypa_net.node_to_name_map().items()}
        adjsum = self.adjacency.sum()
        for u,v,xival in zip(xicoo.row, xicoo.col, xicoo.data):
            source, target = reverse_name_dict[u],reverse_name_dict[v]
            add_edge(u, v, xival, xisum, adjsum, reverse_name_dict)

    def compute_hypa(self, obs_freq, xi, total_xi, total_observations, log_p=True):
        """
        Compute hypa score using ghypernet in R.
        If r2py fails:
            from scipy.stats import hypergeom
            return hypergeom.cdf(obs_freq, total_xi, xi, total_observations)
        """
        hy = Hypergeometric(total_observations, total_xi - total_observations, xi)

        if log_p:
            return logcdf(hy, obs_freq)
        else:
            return cdf(hy, obs_freq)
