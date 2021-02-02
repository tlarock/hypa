import importlib
import numpy as np
import scipy.sparse as sp
import pathpy as pp
from random import shuffle
from .computexi import computeXiHigherOrder, fitXi, xi_matrix

class Hypa:
    '''
    Class for computing hypa scores on a DeBruijn graph given pathway data.
    '''
    def __init__(self, implementation):
        """
        Initialize class with pathpy.paths object.

        parameters
        -----------

        paths: Paths
            Paths object containing the pathway data.
        """
        assert implementation in ['julia', 'rpy2', 'scipy'], "Invalid implementation."

        self.implementation = implementation

        # only import the relevant distribution function to be used in compute_hypa
        if self.implementation == 'julia':
            global Hypergeometric, cdf, logcdf, rand
            from julia.Distributions import Hypergeometric, cdf, logcdf, rand
        elif self.implementation == 'rpy2':
            ## import ghypernet from R
            import rpy2.robjects as ro
            import rpy2.robjects.numpy2ri
            from rpy2.robjects.packages import importr
            rpy2.robjects.numpy2ri.activate()
            self.rphyper = ro.r['phyper']
            self.rrhyper = ro.r['rhyper']
        elif self.implementation == 'scipy':
            global hypergeom
            from scipy.stats import hypergeom


    @classmethod
    def from_paths(cls, paths, k, implementation='julia', **kwargs):
        self = cls(implementation=implementation)
        self.paths = paths
        self.construct_hypa_network(k=k, **kwargs)

        return self

    @classmethod
    def from_graph_file(cls, input_file, implementation='julia', xitol=1e-2, sparsexi=True, verbose=True):
        ''' Read a file to initialize the hypa object. We assume
            the file represents a kth order graph and each line
            is of the form
                        u_1,u_2,...,u_{k+1},freq
            Note that the nodes and edges must be parsed from
            this representation.
        '''
        self = cls(implementation=implementation)
        self.paths = None
        self.hypa_net = pp.Network(directed=True)
        first_order = pp.Network(directed=True)

        if verbose:
            print("Reading file.")
        with open(input_file, 'r') as fin:
            ## Can I sort the input so that I can always compute Xi? Could put M at the top of a file
            for line in fin:
                line = line.strip()
                line_list = line.split(',')
                path = line_list[0:-1]
                ## Inferring k
                k = len(path)-1
                freq = int(line_list[-1])
                u, v = ','.join(path[0:k]), ','.join(path[1:])
                self.hypa_net.add_edge(u, v, weight=freq)
                for i in range(1, len(path)):
                    edge = (path[i-1], path[i])
                    if edge not in first_order.edges:
                        first_order.add_edge(path[i-1], path[i])

        if verbose:
            print(f"Computing the k={k} order Xi")
        self.adjacency = self.hypa_net.adjacency_matrix()
        for node in self.hypa_net.nodes:
            source = node
            ## If this node has 0 outweight, it will have all 0 xi so can be ignored
            if self.hypa_net.nodes[source]['outweight'] == 0:
                continue

            node_as_path = node.split(',')
            fo_neighbors = first_order.successors[node_as_path[-1]]
            for neighbor in fo_neighbors:
                ## ToDo separator in split
                target = ','.join(node_as_path[1:]) + f',{neighbor}'
                ## If target is not a node or has 0 inweight, it will have 0 xi so can be ignored
                if target in self.hypa_net.nodes:
                    # Splitting in to 2 lines to avoid defaultdict issue
                    if self.hypa_net.nodes[target]['inweight'] > 0:
                        xi_val = self.hypa_net.nodes[source]['outweight'] * self.hypa_net.nodes[target]['inweight']
                        self.hypa_net.edges[(source, target)]['xival'] = xi_val
                        if 'weight' not in self.hypa_net.edges[(source, target)]:
                            self.hypa_net.edges[(source, target)]['weight'] = 0.0

        self.Xi = xi_matrix(self.hypa_net)
        self.Xi = fitXi(self.adjacency, self.Xi, sparsexi=sparsexi, tol=xitol, verbose=verbose)

        if verbose:
            print("Computing HYPA scores")
        reverse_name_dict = {val:key for key,val in self.hypa_net.node_to_name_map().items()}
        adjsum = self.adjacency.sum()
        xisum = self.Xi.sum()
        for u,v,xival in zip(self.Xi.row, self.Xi.col, self.Xi.data):
            source, target = reverse_name_dict[u], reverse_name_dict[v]
            pval = self.compute_hypa(self.adjacency[u,v], xival, xisum, adjsum, log_p=True)
            if xival > 0:
                try:
                    self.hypa_net.edges[(source, target)]['pval'] = pval
                    self.hypa_net.edges[(source, target)]['xival'] = xival
                except Exception as e:
                    attr = {'weight': 0.0, 'pval':pval, 'xi':xival}
                    self.hypa_net.add_edge(source, target, **attr)

        return self

    def initialize_xi(self, k=2, sparsexi=True, redistribute=True, xifittol=1e-2, verbose=True):
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
        verbose: logical
            If True, print out details of what is happening.


        """
        if verbose:
            print('Computing the k={} order Xi...'.format(k))

        ## Assign k
        self.k = k

        ## Compute Xi. Also returns a network object.
        self.Xi, self.hypa_net = computeXiHigherOrder(self.paths, k=self.k, sparsexi=sparsexi)
        self.adjacency = self.hypa_net.adjacency_matrix()

        if redistribute:
            if verbose:
                print('Fitting Xi...')

            # Fit the Xi matrix to preserve expected in/out weights
            self.Xi = fitXi(self.adjacency, self.Xi, sparsexi=sparsexi, tol=xifittol, verbose=verbose)
        self.adjacency = self.hypa_net.adjacency_matrix()


    def construct_hypa_network(self, k=2, log=True, sparsexi=True, redistribute=True, xifittol=1e-2, verbose=True):
        """
        Function to compute the significant pathways from a Paths object.

        parameters
        -----------

        paths: Paths
            Paths object containing the pathway data.
        order: int
            The order at which the significance should be computed.
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
            source, target = reverse_name_dict[u],reverse_name_dict[v]
            pval = self.compute_hypa(self.adjacency[u,v], xival, xisum, adjsum, log_p=True)
            if xival > 0:
                try:
                    self.hypa_net.edges[(source, target)]['pval'] = pval
                    ## xival may have been updated by fitXi, make sure it is set correctly here
                    self.hypa_net.edges[(source, target)]['xival'] = xival
                except Exception as e:
                    attr = {'weight': 0.0, 'pval':pval, 'xi':xival}
                    self.hypa_net.add_edge(source, target, **attr)
            return 1


        ## assign k
        self.k = k

        ## create network and Xi matrix
        self.initialize_xi(k=self.k, sparsexi=sparsexi, redistribute=redistribute, xifittol=xifittol, verbose=verbose)

        xisum = self.Xi.sum()
        xicoo = sp.coo_matrix(self.Xi)

        node_name_map = self.hypa_net.node_to_name_map()
        reverse_name_dict = {val:key for key,val in node_name_map.items()}
        adjsum = self.adjacency.sum()
        for u,v,xival in zip(xicoo.row, xicoo.col, xicoo.data):
            if xival > 0:
                add_edge(u, v, xival, xisum, adjsum, reverse_name_dict)
            elif self.adjacency[u,v] == 0:
                del self.hypa_net.edges[(reverse_name_dict[u],reverse_name_dict[v])]

    def compute_hypa(self, obs_freq, xi, total_xi, total_observations, log_p=True):
        """
        Compute hypa score using appropriate implementation.
        """
        if self.implementation == 'julia':
            hy = Hypergeometric(total_observations, total_xi - total_observations, xi)
            if log_p:
                return logcdf(hy, obs_freq)
            else:
                return cdf(hy, obs_freq)
        elif self.implementation == 'rpy2':
            return self.rphyper(obs_freq, xi, total_xi-xi, total_observations, log_p=log_p)[0]
        elif self.implementation == 'scipy':
            if log_p:
                return hypergeom.logcdf(obs_freq, total_xi, xi, total_observations)
            else:
                return hypergeom.cdf(obs_freq, total_xi, xi, total_observations)

    def draw_sample(self, implementation=None, seed=None):
        r"""
        Draw a sample from the hypergeometric ensemble.

        Drawing a sample is implemented as sampling a weight
            for each edge. Sampled weights are stored in the
            self.hypa_net.edges dictionary with the key
            'sampled_weight'.

        NOTE: Drawing new sample overwrites old values of 'sampled_weight'

        Currently implemented for Juilia and rpy2.

        Implementations inspired by numpy.random.Generator.multivariate_hypergeometric ("marginals" option):
            https://numpy.org/devdocs/reference/random/generated/numpy.random.Generator.multivariate_hypergeometric.html

        Parameters
        --------

        Returns
        --------

        """
        if implementation is None:
            implementation = self.implementation

        if implementation == 'julia':
            self.draw_sample_julia()
        elif implementation ==  'rpy2':
            self.draw_sample_rpy2()
        elif implementation == 'scipy':
            self.draw_sample_numpy(seed)

    def draw_sample_julia(self):
        total_xi = self.Xi.sum()
        ## Sample once per existing edge
        num_samples = self.adjacency.sum()
        xi_accum = 0
        ## Loop over the edges
        for i, edge in enumerate(self.hypa_net.edges):
            if num_samples < 1:
                break
            xi = self.hypa_net.edges[edge]['xival']
            xi_accum += xi
            #if i < len(edges_list) - 1:
            if i < len(self.hypa_net.edges) - 1:

                # Hypergeometric distribution for a population with s successes and f failures, and a sequence of n trials.
                #Hypergeometric(s, f, n)
                hy = Hypergeometric(xi, total_xi - xi_accum, num_samples)
                sample = rand(hy)
                self.hypa_net.edges[edge]['sampled_weight'] = sample
            else:
                sample = int(num_samples)
                self.hypa_net.edges[edge]['sampled_weight'] = sample

            num_samples -= sample

    def draw_sample_rpy2(self):
        total_xi = self.Xi.sum()
        num_samples = self.adjacency.sum()
        xi_accum = 0

        edges_list = list(self.hypa_net.edges)
        shuffle(edges_list)
        for i, edge in enumerate(edges_list):
            if num_samples < 1:
                break
            xi = self.hypa_net.edges[edge]['xival']
            xi_accum += xi
            if i < len(edges_list) - 1:
                sample = self.rrhyper(1, xi, total_xi-xi_accum, num_samples)[0]
                self.hypa_net.edges[edge]['sampled_weight'] = sample
            else:
                sample = int(num_samples)
                self.hypa_net.edges[edge]['sampled_weight'] = sample

            num_samples -= sample

    def draw_sample_numpy(self, seed=None):
        ## Read xi values in order of hypa_net.edges
        colors = [int(data['xival']) for _, data in self.hypa_net.edges.items() if data['xival'] > 0]
        nsamples = int(self.adjacency.sum())
        gen = np.random.Generator(np.random.PCG64(seed))
        variates = gen.multivariate_hypergeometric(colors, nsamples, method='count')
        ## Write sampled_weights in order of hypa_net.edges
        for i, edge in enumerate(self.hypa_net.edges):
            self.hypa_net.edges[edge]['sampled_weight'] = variates[i]
