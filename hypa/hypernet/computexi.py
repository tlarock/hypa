import numpy as np
import scipy.sparse as sp
import pathpy as pp

def computeXiHigherOrder(paths, k = 2, sparsexi=False, noxi=False):
    """
    Compute the Xi matrix for higher order networks, starting from a HigherOrderNetwork pathpy object.
    ## in the higher order case
    # (1) Self-loops are completely disallowed
    # (2) Only viable edges count towards the probablity

    """
    separator = paths.separator

    ## the weighted xi network (could just be a matrix, not totally necessary to have a network)
    network = pp.Network(directed=True)

    higher_order = pp.HigherOrderNetwork(paths, k, separator=separator)
    ## The network to return, which has the true observation frequencies as weights
    ## this way we know that the Xi and adjacency matrices should have the same non-zero entries
    return_network = pp.Network(directed=True)

    ## generate all possible paths from the first order network
    first_order = pp.HigherOrderNetwork(paths, k=1, separator=separator)
    possible_paths = pp.HigherOrderNetwork.generate_possible_paths(first_order, k)

    if noxi:
        xi_const = 0
        edges_sofar = 0

    for path in possible_paths:
        source, target = higher_order.path_to_higher_order_nodes(path)

        if (source,target) not in network.edges:
            ## xi computation
            xi_val = higher_order.nodes[source]['outweight'].sum() * higher_order.nodes[target]['inweight'].sum()
            if xi_val == 0:
                continue
            if noxi:
                network.add_edge(source, target)
                edges_sofar += 1
                xi_const += (xi_val - xi_const) / edges_sofar
            else:
                network.add_edge(source, target, weight=xi_val)

            ## add the total observations of this path to the return network
            observations = paths.paths[k][path].sum()
            return_network.add_edge(source, target, weight=float(observations))
    if noxi:
        xi_const = np.round(xi_const)
        for e in network.edges:
            network.edges[e]['weight'] = xi_const

    if sparsexi:
        xi = network.adjacency_matrix(weighted=True).tocoo()
    else:
        xi = network.adjacency_matrix(weighted=True).toarray()

    return xi, return_network



def fitXi(adj, xi_input, tol=1e-2, sparsexi=False, verbose=False):
    """
    python version of new r code to fix xi
    """
    def xifix_row(m, xi, degs):
        xi_sum = xi.sum()

        # uniformly increase sum(xi) to m^2
        xi = xi * m**2/xi_sum
        xi_sum = xi.sum()

        # compute the ratio between observed degrees and
        # expected degrees (rowwise)
        exp_degs = (xi/xi_sum*m).sum(1)
        nonzero_ids = exp_degs != 0

        ratio = np.zeros(len(exp_degs))
        ratio[nonzero_ids] = degs[nonzero_ids]/exp_degs[nonzero_ids]

        # increment each column of xi by the computed ratio
        # this results in a xi matrix with the correct number
        # of balls (m^2 == sum(xi)) and degree preserved
        # rowwise
        xi = (xi.T * ratio).T

        # return the xi rounded to integers
        return np.round(xi)

    ## pass adjacency and non-corrected xi. if verbose=T,
    ## print root mean squared error (rmse)

    if not sparsexi and sp.issparse(xi_input):
        xi = xi_input.toarray()
    else:
        xi = xi_input

    # the total number of edges
    m = adj.sum()
    ## if Scipy
    if sparsexi:
        indegs = np.array(adj.sum(0))
        outdegs = np.array(adj.sum(1))
        indegs = np.reshape(indegs, max(indegs.shape))
        outdegs = np.reshape(outdegs, max(outdegs.shape))
    else:
        indegs = adj.sum(0)
        outdegs = adj.sum(1)

    # the baseline rmse
    xi_sum = xi.sum()
    val = ((xi/xi_sum*m).sum(1) - outdegs)
    rmse = np.sqrt( ( ((xi/xi_sum*m).sum(axis=1) - outdegs)**2 ).sum() )/2 + np.sqrt( ( ((xi/xi_sum*m).sum(axis=0) - indegs)**2 ).sum() )/2
    rmseold = rmse.copy()
    if verbose:
       print('rmse =',rmse)

    # loop to alternatively fix rows and columns
    while True:
        # fix row degrees
        xi = xifix_row(m = m, xi = xi, degs = outdegs)
        # fix column degrees applying xifit_row to transposed xi
        xi = ( xifix_row(m = m, xi = xi.T, degs = indegs) ).T

        # compute rmse
        xi_sum = xi.sum()
        rmse = np.sqrt( ( ((xi/xi_sum*m).sum(1) - outdegs)**2 ).sum() )/2 + np.sqrt( ( ((xi/xi_sum*m).sum(0) - indegs)**2 ).sum() )/2
        if verbose:
            print('rmse =',rmse)

        # break when no improvements in rmse or rmse less than tol
        if rmseold <= rmse or rmse < tol :
             break
        # reassign rmse for new round
        rmseold = rmse.copy()

    if sp.issparse(xi_input):
        xi = sp.coo_matrix(xi)

    # return new xi
    return xi

