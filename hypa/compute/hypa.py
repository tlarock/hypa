import numpy as np
import scipy.sparse as sp
import pathpy as pp

## import hypernets from R
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

## import my code to work with hypernets
from ..hypernet import computeXiHigherOrder, fitXi, ghype

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
        self.initialize_R(ghype_r)

    def initialize_R(self, ghype_r):
        '''Initialize rpy2 functions'''
        rpy2.robjects.numpy2ri.activate()

        self.hypernets = importr('hypernets')
        self.rphyper = ro.r['phyper']
        self.randomgraph = ro.r['RandomGraph']
        self.ghype_r = ghype_r


    def construct_hypa_network(self, k=2, log=True, sparsexi=True, redistribute=True, xifittol=1e-2, constant_xi=False, verbose=True):
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
        if verbose:
            print('Computing the k={} order Xi...'.format(k))

        ## Compute Xi TODO assuming sparse matrix here
        self.Xi, self.network = computeXiHigherOrder(self.paths, k=k, sparsexi=sparsexi, noxi=constant_xi)
        self.adjacency = self.network.adjacency_matrix()

        if redistribute and not constant_xi:
            if verbose:
                print('Fitting Xi...')

            ## TODO again assuming sparse matrix
            self.Xi = fitXi(self.adjacency, self.Xi, sparsexi=sparsexi, tol=xifittol, verbose=verbose)

        reverse_name_dict = {val:key for key,val in self.network.node_to_name_map().items()}

        self.adjacency = self.network.adjacency_matrix()


        ## construct the network of underrepresented pathways
        self.hypa_net = pp.Network(directed=True)
        adjsum = self.adjacency.sum()
        xisum = self.Xi.sum()
        xicoo = sp.coo_matrix(self.Xi)
        for u,v,xival in zip(xicoo.row, xicoo.col, xicoo.data):
        # for u,v in edge_likelihood_sortidx:
            source, target = reverse_name_dict[u],reverse_name_dict[v]
            pval = self.compute_hypa(self.adjacency[u,v], xival, xisum, adjsum, log_p=True)
            if xival == 0:
                continue

            try:
                attr = {'weight': self.network.edges[(source, target)]['weight'], 'pval':pval, 'xi':xival}
            except Exception as e:
                attr = {'weight': 0.0, 'pval':pval, 'xi':xival}
            self.hypa_net.add_edge(source, target, **attr)


    def compute_hypa(self, obs_freq, xi, total_xi, total_observations, log_p=True):
        """
        Compute hypa score using hypernets in R.
        """
        return self.rphyper(obs_freq, xi, total_xi-xi, total_observations, log_p=log_p)[0]


    def draw_sample(self,k=2):
        if self.ghype_r is None:
            adj = self.adjacency.toarray()
            adjr = ro.r.matrix(adj, nrow=adj.shape[0], ncol=adj.shape[1])
            ro.r.assign('adj', adjr)
            ## Use constant omega
            omega = np.ones(adj.shape)
            self.ghype_r = self.hypernets.ghype(adj, directed=True, selfloops=False, xi=self.Xi.toarray(), omega=omega)

        sampled_adj = self.randomgraph(1, self.ghype_r, m=self.adjacency.sum(), multinomial=False)

        return np.array(sampled_adj)

