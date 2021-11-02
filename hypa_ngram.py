import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix
from computexi import fitXi
from scipy.stats import hypergeom


def hypa_from_ngram(input_file, k, frequency=False, verbose=True):
    '''
    Accepts an ngram filename and integer k and computes a kth-order
    HON from ngram, including unobserved but possible edges. This
    is necessary for correctly computing the xi matrix in the
    next step.

    Parameters
    ---------
    input_file (str): path to the ngram file
    k (int): order of the HON
    frequency (bool): if True, ngram file rows
                    end with a value to be interpreted
                    as the frequency of the path
    verbose (bool): if True, print more stuff
    '''
    # Read ngram file
    hypa_net = nx.DiGraph()
    first_order = nx.DiGraph()

    lines_read = 0
    interval_lines = 0
    interval = 100_000
    # Read the input
    with open(input_file) as fin:
        for line in fin:
            lines_read += 1
            interval_lines += 1
            if interval_lines == interval and verbose:
                print(f'{lines_read} lines read.', flush=True)
                interval_lines = 0
            # Strip trailing whitespace
            line = line.strip().split(',')

            # parse path
            if frequency:
                path = line[0:-1]
                freq = float(line[-1])
            else:
                path = line
                freq = 1

            # Add first order edges (always)
            for i in range(1, len(path)):
                edge = (path[i-1], path[i])
                if edge not in first_order.edges:
                    first_order.add_edge(path[i-1], path[i])

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

    return hypa_net


def construct_xi(hypa_net, xitol=1e-2, verbose=False):
    '''
    Construct the xi matrix from a hypa_net.

    Note: Relies on fitXi in computexi.py.
    '''
    N = len(hypa_net.nodes())
    adj = dok_matrix((N, N))
    xi = dok_matrix((N, N))
    # compute Xi for every edge in hypa_net
    node_to_idx = dict()
    idx = 0
    for u, v, edat in hypa_net.edges(data=True):
        xival = hypa_net.out_degree(u, weight='weight') *\
                hypa_net.in_degree(v, weight='weight')
        if u not in node_to_idx:
            node_to_idx[u] = idx
            idx += 1
        if v not in node_to_idx:
            node_to_idx[v] = idx
            idx += 1

        xi[node_to_idx[u], node_to_idx[v]] = xival
        edat['xival'] = xival
        adj[node_to_idx[u], node_to_idx[v]] = edat['weight']

    xi = fitXi(adj, xi, tol=xitol, sparsexi=True, verbose=verbose)
    xi = xi.tocsr()
    for u, v, edat in hypa_net.edges(data=True):
        edat['xival_orig'] = float(edat['xival'])
        edat['xival'] = xi[node_to_idx[u], node_to_idx[v]]
    return adj, xi


def compute_pvals(hypa_net, adj, xi, observed_only=True, remove_unobserved=True):
    '''
    Compute pvalues for the edges in hypa_net. If observed_only is True,
    only compute pvalues for edges with weight > 0.
    '''
    # loop over all edges in hypa_net
    xisum = xi.sum()
    adjsum = adj.sum()
    edges_to_remove = []
    for u, v, edat in hypa_net.edges(data=True):
        if observed_only and edat['weight'] == 0:
            if remove_unobserved:
                edges_to_remove.append((u, v))
            continue
        pval = compute_hypa(edat['weight'], edat['xival'], xisum, adjsum)
        edat['log-pval'] = pval
        edat['pval'] = np.exp(pval)

    if len(edges_to_remove) > 0:
        hypa_net.remove_edges_from(edges_to_remove)


def compute_hypa(obs_freq, xi, total_xi, total_observations, log_p=True):
    """
    Compute hypa score. Only using the scipy implementation here.
    """
    if log_p:
        return hypergeom.logcdf(obs_freq, total_xi, xi, total_observations)
    else:
        return hypergeom.cdf(obs_freq, total_xi, xi, total_observations)
