import numpy as np
import scipy.sparse as sp
import pathpy as pp


def xi_matrix(network):
    """Returns a sparse xi matrix of the higher-order network. Unless transposed
    is set to true, the entry corresponding to a directed link s->t is stored in row s and
    column t and can be accessed via A[s,t].

    Parameters
    ----------

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
        data.append(e['xival'])

        if not network.directed and t != s:
            row.append(node_to_coord[t])
            col.append(node_to_coord[s])
            data.append(e['xival'])

    shape = (network.ncount(), network.ncount())
    A = sp.coo_matrix((data, (row, col)), shape=shape).tocsr()

    return A

def computeXiHigherOrder(paths, k = 2, sparsexi=False):
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

    """
    separator = paths.separator

    ## the weighted xi network (could just be a matrix, not totally necessary to have a network)
    network = pp.Network(directed=True)

    ## ToDo: Rather than using 2 pathpy constructors, could write code to
    ## generate both of the below networks in a single loop over paths

    ## generate higher order network, giving us nodes and (non-zero) edges
    higher_order = pp.HigherOrderNetwork(paths, k, separator=separator)

    ## generate the first order network, we will use this for generating possible neighbors
    first_order = pp.HigherOrderNetwork(paths, k=1, separator=separator)

    for node in higher_order.nodes:
        source = node
        ## If this node has 0 outweight, it will have all 0 xi so can be ignored
        if higher_order.nodes[source]['outweight'].sum() == 0:
            continue

        node_as_path = node.split(separator)
        fo_neighbors = first_order.successors[node_as_path[-1]]
        for neighbor in fo_neighbors:
            ## ToDo separator in split
            if k > 1:
                target = ','.join(node_as_path[1:]) + f',{neighbor}'
            else:
                target = neighbor

            ## If target is not a node or has 0 inweight, it will have 0 xi so can be ignored
            if target in higher_order.nodes:
                ## Splitting in to 2 conditionals to avoid defaultdict issue
                if higher_order.nodes[target]['inweight'].sum() > 0:
                    if (source,target) not in network.edges:
                        ## xi computation
                        xi_val = higher_order.nodes[source]['outweight'].sum() * higher_order.nodes[target]['inweight'].sum()
                        if xi_val == 0:
                            continue

                        ## add the total observations of this path to the return network
                        path = tuple(source.split(separator)) + tuple([target.split(separator)[-1]])
                        observations = paths.paths[k][path].sum()
                        network.add_edge(source, target, weight=observations, xival=xi_val)


    if sparsexi:
        xi = xi_matrix(network).tocoo()
    else:
        xi = xi_matrix(network).toarray()

    return xi, network


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

