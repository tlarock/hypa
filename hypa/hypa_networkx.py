import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix
from scipy.stats import hypergeom

from .computexi import fitXi

class HypaNX():
    def __init__(self, k, input_file=None, paths=None, xitol=1e-2,
                 frequency=False, verbose=True, log_p=False,
                 compute_scores=True, construct_xi=True,
                 check_selfloops=False):
        '''
        Accepts a path dataset in one of 3 formats as well as
        an integer k. Computes a kth-order HON from ngram,
        optionally including unobserved but possible edges.
        (necessary for correctly computing the xi matrix).

        Input formats:
            1. input_file (str) that points to a csv where each
                row is a path. If frequency = True, each line
                in input_file ends with a numerical value
                indicating how many times the path was observed.
            2. paths (list or dict) containing path data. If a list,
                each  path is added with frequency 1. If a dictionary,
                values are assumed to be frequencies of paths.

        Parameters
        ---------
        input_file (str): path to the ngram file
        k (int): order of the HON
        frequency (bool): if True, ngram file rows
                        end with a value to be interpreted
                        as the frequency of the path
        log_p (bool): if True, comput probabilities in
                        log space. WARNING: for some
                        reason, _logpmf is EXTREMELY
                        slow. Recommend setting this param
                        to false whenever numerically possible.
        verbose (bool): if True, print more stuff
        '''
        self.implementation = 'scipy'
        if input_file is not None:
            self.input_file = input_file
            ngram = True
        elif paths is not None:
            dict_flag = False
            if isinstance(paths, dict):
                dict_flag = True
            self.paths = paths
            ngram = False
        else:
            assert input_file or paths, "Need to specify one of input_file or paths."

        self.k = k
        self.frequency = frequency
        self.verbose = verbose
        self.xitol = xitol
        self.log_p = log_p
        self.check_selfloops = check_selfloops
        if verbose:
            print("Constructing HON.")

        if ngram:
            self.hypa_from_ngram()
        else:
            self.hypa_from_list(dict_flag)

        if construct_xi:
            if verbose:
                print("Constructing xi.")

            self.construct_xi()

        if compute_scores:
            if verbose:
                print("Computing pvals.")
            self.compute_pvals()

    def hypa_from_list(self, dict_flag):
        def add_first_order(first_order, path, freq):
            for i in range(1, len(path)):
                u, v = path[i-1], path[i]
                if not first_order.has_edge(u, v):
                    first_order.add_edge(u, v, weight=freq)
                else:
                    first_order.edges[(u, v)]['weight'] += freq

        hypa_net = nx.DiGraph()
        first_order = nx.DiGraph()
        k = self.k
        for path in self.paths:
            if dict_flag:
                freq = self.paths[path]
            else:
                freq = 1

            path = list(map(str, path))
            if self.check_selfloops:
                path = remove_selfloops(path)

            add_first_order(first_order, path, freq)
            # Skip a path if its length is less than k
            k = self.k
            if len(path)-1 < k:
                continue

            # Add all nodes and edges
            for i in range(0, len(path)-k):
                u, v = ','.join(path[i:i+k]), ','.join(path[i+1:i+k+1])
                if (u, v) in hypa_net.edges:
                    hypa_net.edges[(u, v)]['weight'] += freq
                else:
                    hypa_net.add_edge(u, v, weight=freq)

        # To compute xi correctly, I need to include
        # all of the possible edges that had 0 frequency.
        edges_to_add = []
        for node in hypa_net.nodes():
            splitnode = node.split(',')
            prefix = splitnode[1:]
            target = splitnode[-1]
            for successor in first_order.successors(target):
                new_node = prefix + [successor]
                new_node_str = ','.join(new_node)
                if (node, new_node_str) not in hypa_net.edges():
                    edges_to_add.append((node, new_node_str, {'weight': 0}))
        hypa_net.add_edges_from(edges_to_add)
        self.hypa_net = hypa_net
        self.first_order = first_order

    def hypa_from_ngram(self):
        # Read ngram file
        hypa_net = nx.DiGraph()
        first_order = nx.DiGraph()

        k = self.k
        lines_read = 0
        interval_lines = 0
        interval = 100_000
        obs_sum = 0
        # Read the input
        with open(self.input_file) as fin:
            for line in fin:
                lines_read += 1
                interval_lines += 1
                if interval_lines == interval and self.verbose:
                    print(f'{lines_read} lines read.', flush=True)
                    interval_lines = 0
                # Strip trailing whitespace
                line = line.strip().split(',')

                # parse path
                if self.frequency:
                    path = line[0:-1]
                    freq = float(line[-1])
                else:
                    path = line
                    freq = 1

                if self.check_selfloops:
                    path = remove_selfloops(path)
                # Add first order edges (always)
                for i in range(1, len(path)):
                    edge = (path[i-1], path[i])
                    if edge not in first_order.edges:
                        first_order.add_edge(edge[0], edge[1], weight=freq)
                    else:
                        first_order.edges[edge]['weight'] += freq

                # Skip a path if its length is less than k
                if len(path)-1 < k:
                    continue

                # Add all nodes and edges
                for i in range(0, len(path)-k):
                    u, v = ','.join(path[i:i+k]), ','.join(path[i+1:i+k+1])
                    if (u, v) in hypa_net.edges:
                        hypa_net.edges[(u, v)]['weight'] += freq
                    else:
                        hypa_net.add_edge(u, v, weight=freq)
                    obs_sum += freq

        # To compute xi correctly, I need to include
        # all of the possible edges that had 0 frequency.
        edges_to_add = []
        for node in hypa_net.nodes():
            splitnode = node.split(',')
            prefix = splitnode[1:]
            target = splitnode[-1]
            for successor in first_order.successors(target):
                new_node = prefix + [successor]
                new_node_str = ','.join(new_node)
                if (node, new_node_str) not in hypa_net.edges():
                    edges_to_add.append((node, new_node_str, {'weight': 0}))
        hypa_net.add_edges_from(edges_to_add)
        self.hypa_net = hypa_net
        self.first_order = first_order

    def construct_xi(self):
        '''
        Construct the xi matrix from a hypa_net.

        Note: Relies on fitXi in computexi.py.
        '''
        N = len(self.hypa_net.nodes())
        adj = dok_matrix((N, N))
        xi = dok_matrix((N, N))
        # compute Xi for every edge in hypa_net
        node_to_idx = dict()
        idx = 0
        for u, v, edat in self.hypa_net.edges(data=True):
            xival = self.hypa_net.out_degree(u, weight='weight') *\
                    self.hypa_net.in_degree(v, weight='weight')
            if u not in node_to_idx:
                node_to_idx[u] = idx
                idx += 1
            if v not in node_to_idx:
                node_to_idx[v] = idx
                idx += 1

            xi[node_to_idx[u], node_to_idx[v]] = xival
            edat['xival'] = xival
            adj[node_to_idx[u], node_to_idx[v]] = edat['weight']

        xi = fitXi(adj, xi, tol=self.xitol,
                   sparsexi=True, verbose=self.verbose)
        xi = xi.tocsr()

        # Remove edges from the graph that have zero xi
        edges_to_remove = []
        for u, v, edat in self.hypa_net.edges(data=True):
            if edat['xival'] > 0:
                edat['xival_orig'] = float(edat['xival'])
                edat['xival'] = xi[node_to_idx[u], node_to_idx[v]]
            else:
                edges_to_remove.append((u, v))

        self.hypa_net.remove_edges_from(edges_to_remove)
        self.adj = adj
        self.xi = xi
        self.node_to_idx = node_to_idx

    def compute_pvals(self):
        '''
        Compute pvalues for the edges in hypa_net.
        '''
        # loop over all edges in hypa_net
        xisum = int(self.xi.sum())
        adjsum = int(self.adj.sum())
        idx_to_node = {idx:node for node, idx in self.node_to_idx.items()}
        rows, cols = self.xi.nonzero()
        for row, col in zip(rows,cols):
            edge = idx_to_node[row], idx_to_node[col]
            if self.log_p:
                self.hypa_net.edges[edge]['log-pval'] = hypergeom.logcdf(self.adj[row,col],
                                                                         xisum,
                                                                         self.xi[row,col],
                                                                         adjsum)
                self.hypa_net.edges[edge]['pval'] = np.exp(self.hypa_edges[edge]['log-pval'])
            else:
                self.hypa_net.edges[edge]['pval'] = hypergeom.cdf(self.adj[row,col],
                                                                  xisum,
                                                                  self.xi[row,col],
                                                                  adjsum)
                self.hypa_net.edges[edge]['log-pval'] = np.log(self.hypa_net.edges[edge]['pval'])

    def draw_sample(self, seed=None):
        xisum = int(self.xi.sum())
        adjsum = int(self.adj.sum())
        idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        rows, cols = self.xi.nonzero()
        for row, col in zip(rows, cols):
            edge = idx_to_node[row], idx_to_node[col]
            assert self.xi[row, col] > 0, f"xi 0 for {row}, {col}, {edge}"
            weight = hypergeom.rvs(xisum, int(self.xi[row, col]), adjsum)
            self.hypa_net.edges[edge]['sampled_weight'] = weight

    def draw_sample_mat(self):
        sample_mat = hypergeom.rvs(self.xi.sum(), self.xi.todense(),
                                   self.adj.sum())
        sampled_graph = nx.from_numpy(sample_mat)
        return sampled_graph

def remove_selfloops(path):
    '''
    Accepts a path and removes any selfloops.
    '''
    i = 1
    end = len(path)
    while i < end:
        if path[i-1] == path[i]:
            del path[i-1]
            end-=1
        else:
            i+=1
    return path
