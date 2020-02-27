import numpy as np
import scipy.sparse as sp
import pathpy as pp


def xi_matrix(network, weighted=True, transposed=False):
    """Returns a sparse xi matrix of the higher-order network. Unless transposed
    is set to true, the entry corresponding to a directed link s->t is stored in row s and
    column t and can be accessed via A[s,t].

    Parameters
    ----------
    weighted: bool
	if set to False, the function returns a binary adjacency matrix.
	If set to True, adjacency matrix entries contain edge weights.
    transposed: bool
	whether to transpose the matrix or not.

    Returns
    -------
    numpy cooc matrix
    """
    row = []
    col = []
    data = []

    edgeC = network.ecount()
    if not network.directed:
        n_self_loops = sum(s == t for (s, t) in network.edges)
        edgeC *= 2
        edgeC -= n_self_loops

    node_to_coord = network.node_to_name_map()
    for (s, t), e in network.edges.items():
        row.append(node_to_coord[s])
        col.append(node_to_coord[t])
        if weighted:
            data.append(e['xival'])
        else:
            data.append(1)

        if not network.directed and t != s:
            row.append(node_to_coord[t])
            col.append(node_to_coord[s])
            if weighted:
                data.append(e['xival'])
            else:
                data.append(1)

    shape = (network.ncount(), network.ncount())
    A = sp.coo_matrix((data, (row, col)), shape=shape).tocsr()

    if transposed:
        return A.transpose()
    return A

def computeXiHigherOrder(higher_order, k = 2, sparsexi=False, constant_xi=False):
    r"""
    Compute the Xi matrix for higher order networks.

    Parameters
    ----------
    paths: pp.Paths
        Paths object
    k: int
        Order to compute the Xi
    sparsexi: logical
        If True, use scipy sparse matrices. Default False (numpy arrays).
    constant_xi: logical
        If True, use a constant xi matrix (null model). Default False. TODO: Better explanation


    Side effect: Unobserved but possible edges will be added with weight 0.
    """
    paths = higher_order.paths
    separator = paths.separator

    ## generate all possible paths from the first order network
    first_order = pp.HigherOrderNetwork(paths, k=1, separator=separator)
    possible_paths = pp.HigherOrderNetwork.generate_possible_paths(first_order, k)

    if constant_xi:
        xi_const = 0
        edges_sofar = 0

    for path in possible_paths:
        source, target = higher_order.path_to_higher_order_nodes(path)

        ## xi computation
        xi_val = higher_order.nodes[source]['outweight'].sum() * higher_order.nodes[target]['inweight'].sum()
        if xi_val == 0:
            continue

        ## add the total observations of this path to the return network
        observations = paths.paths[k][path].sum()
        if 'weight' not in higher_order.edges[(source,target)]:
            higher_order.edges[(source,target)]['weight'] = np.array([0., 0.])

        if constant_xi:
            edges_sofar += 1
            xi_const += (xi_val - xi_const) / edges_sofar
        else:
            higher_order.add_edge(source, target, xival=xi_val)


    if constant_xi:
        xi_const = np.round(xi_const)
        for e in higher_order.edges:
            higher_order.edges[e]['xival'] = xi_const

    if sparsexi:
        xi = xi_matrix(higher_order, weighted=True).tocoo()
    else:
        xi = xi_matrix(higher_order, weighted=True).toarray()

    return xi


def xifix_row(m, xi, degs):
    xi_sum = xi.sum()

    # uniformly increase sum(xi) to m^2
    xi = xi * m**2/xi_sum
    xi_sum = xi.sum()

    # compute the ratio between observed degrees and
    # expected degrees (rowwise)
    exp_degs = np.array((xi/xi_sum*m).sum(axis=1))
    exp_degs = exp_degs.reshape(max(exp_degs.shape))
    nonzero_ids = exp_degs != 0
    nonzero_ids = nonzero_ids.reshape(max(nonzero_ids.shape,))
    ratio = np.zeros(len(exp_degs))
    ratio[nonzero_ids] = degs[nonzero_ids]/exp_degs[nonzero_ids]
    # increment each column of xi by the computed ratio
    # this results in a xi matrix with the correct number
    # of balls (m^2 == sum(xi)) and degree preserved
    # rowwise
    if not sp.issparse(xi):
        xi = (xi.transpose() * ratio).transpose()
    else:
        xi = (xi.transpose().multiply(ratio)).transpose()

    # return the xi rounded to integers
    return np.round(xi)

def compute_rmse(indegs, outdegs, xi, xi_sum, m):
    val = ((xi/xi_sum*m).sum(axis=1) - outdegs)
    out_sum = np.array((xi/xi_sum*m).sum(axis=1))
    out_sum = out_sum.reshape(max(out_sum.shape))
    in_sum = np.array((xi/xi_sum*m).sum(axis=0))
    in_sum = in_sum.reshape(max(in_sum.shape))
    rmse = np.sqrt( ( (out_sum - outdegs)**2 ).sum() )/2 + np.sqrt( ( (in_sum - indegs)**2 ).sum())/2
    return rmse

def fitXi(adj, xi_input, tol=1e-2, sparsexi=False, verbose=False):
    """
    python version of new r code to fix xi
    """
    ## pass adjacency and non-corrected xi. if verbose=T,
    ## print root mean squared error (rmse)

    if (not sparsexi) and sp.issparse(xi_input):
        xi = xi_input.toarray()
    else:
        xi = xi_input

    # the total number of edges
    m = adj.sum()
    ## if Scipy
    indegs = np.array(adj.sum(0))
    outdegs = np.array(adj.sum(1))
    indegs = np.reshape(indegs, max(indegs.shape))
    outdegs = np.reshape(outdegs, max(outdegs.shape))

    # the baseline rmse
    xi_sum = xi.sum()
    rmse = compute_rmse(indegs, outdegs, xi, xi_sum, m)
    rmseold = rmse.copy()
    if verbose:
       print('rmse =',rmse)

    # loop to alternatively fix rows and columns
    while True:
        # fix row degrees
        xi = xifix_row(m = m, xi = xi, degs = outdegs)
        # fix column degrees applying xifit_row to transposed xi
        xi = ( xifix_row(m = m, xi = xi.transpose(), degs = indegs) ).transpose()

        # compute rmse
        xi_sum = xi.sum()
        rmse = compute_rmse(indegs, outdegs, xi, xi_sum, m)
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

