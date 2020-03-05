import random
import pickle
import numpy as np
from collections import defaultdict
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle
import pathpy as pp
from pathpy.algorithms.random_walk import generate_walk

import hypa

from sklearn import metrics

def create_truth(pnet, abundance=0.3, concentrate_all = True):
    truth_over = []
    ## For every edge
    for e, nei in pnet.successors.items():
        ## Decide if it should be marked anomalous
        if len(nei) < 2 or random.random()<abundance:
            continue

        neighbor_weights = [(t,pnet.edges[(e,t)]['weight']) for t in nei]
        neighbor_weights = sorted(neighbor_weights, key=lambda r: r[1].sum())

        # leave only one with all the weights
        if concentrate_all:
            tokeep = neighbor_weights[-1]
            total_weight = 0
            ## for all of the neighbors except tokeep
            for t,w in neighbor_weights[:-1]:
                total_weight += w
                ## Set the weight to 0
                pnet.edges[(e,t)]['weight'] = np.array([0.0, 0.0])

            ## Set weight of tokeep edge to total weight
            pnet.edges[(e,tokeep[0])]['weight'] += total_weight
            pnet.edges[(e,tokeep[0])]['truth'] = 'over'

            truth_over.append(honwalk2firstwalk( (e,tokeep[0]) ))

        else:
            # make the max under-rep by giving all its weight to another
            # then set its weight to 0 and mark "under"
            # mark the other "over"
            to_under = neighbor_weights[-1]
            pnet.edges[(e,to_under[0])]['weight'] = np.array([0.,0.])
            pnet.edges[(e,to_under[0])]['truth'] = 'under'

            to_over = neighbor_weights[-2]
            pnet.edges[(e,to_over[0])]['weight'] += to_under[1]
            pnet.edges[(e,to_over[0])]['truth'] = 'over'

            truth_over.append(honwalk2firstwalk( (e,to_over[0]) ))

    return truth_over

def highV2lowE(v, sep=','):
    """Takes a k-order node
    Returns the corresponding k-1 order edge
    """
    pathe = v.split(sep)
    s = sep.join(pathe[:-1])
    t = sep.join(pathe[1:])
    return s,t

def honwalk2firstwalk(honvseq, sep=','):
    """Takes a k-order node
    Returns the corresponding path.
    """
    k1walk = honvseq[0].split(sep)
    for hv in honvseq[1:]:
        k1walk.append(hv.split(sep)[-1])
    return k1walk


def compute_roc(pnets, plot=True, output=None, method='hypa', alpha=0.5):
    """
    Compute Reciever Operating Characteristic for a given network
    """
    assert method in ['hypa', 'fbad'], \
            "method must be one of hypa or fbad not {}".format(method)

    k = max(list(pnets.keys()))
    auc_k = []

    for _k in range(1,k+1):
        pickle.dump(pnets[_k].edges, open('edges_pnets_{}.pickle'.format(_k), 'wb'))
        if method == 'fbad':
            edge_weights = [d['weight'] for _,d in pnets[_k].edges.items()]
            mean = np.mean(edge_weights)
            std = np.std(edge_weights)

        y_true = []
        y_score = []

        for e,d in pnets[_k].edges.items():
            if 'truth' in d.keys() and d['truth'] =='over':
                y_true.append(1)
            else:
                y_true.append(0)

            if method == 'hypa':
                y_score.append(np.exp(d['pval']))
            elif method == 'fbad':
                if d['weight'].sum() > (mean + std*alpha):
                    y_score.append(1.0)
                else:
                    y_score.append(0.0)

        #assert sum(y_true) > 0, "no positives in y_true, wtf?\n_k: {}\nEdges:{}".format(_k, pnets[_k].edges)
        #if _k == 5:
        #    print(_k)
        #    print(pnets[_k].edges)
        fpr, tpr, pthr = metrics.roc_curve(y_true, y_score, drop_intermediate=True)

        auc_k.append([_k, metrics.auc(fpr, tpr)])

        if plot:
            plt.plot(fpr, tpr, '.-', label='k={}'.format(_k))

    if plot:
        plt.plot((0,1), (0,1), 'k--')
        plt.legend(loc=6, bbox_to_anchor=(1.,0.5), title="HYPA order")
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.tight_layout()
        if output is not None:
            plt.savefig(output)
    return auc_k



def path_from_hone(e, sep=','):
    return e[0].split(sep) + [e[1].split(sep)[-1]]



def issubpath(subpath, fullpath):
    try:
        l = len(subpath)
    except TypeError:
        l = 1
        subpath = type(fullpath)((subpath,))

    for i in range(len(fullpath)-l+1):
        if fullpath[i:i+l] == subpath:
            return True
    return False



def honseq2prevseq(honvseq, sep=','):
    """
    """
    prevseq = list(highV2lowE(honvseq[0]))
    for hv in honvseq[1:]:
        prevseq.append(highV2lowE(hv)[-1])
    return prevseq


def mark_truth(pnet, truth_over=[], truth_under=[]):
    '''
    '''
    def mark_longer_edge(e, path, true_list, flag):
        for subpath in true_list:
            if issubpath(subpath, path):
                pnet.edges[e]['truth'] = flag

    def mark_shorter_edge(e, path, true_list, flag):
        for trupath in true_list:
            if issubpath(path, trupath):
                pnet.edges[e]['truth'] = flag


    for e in pnet.edges.keys():
        fullpath = path_from_hone(e)

        if len(truth_over)>0:
            if len(fullpath) >= len(truth_over[0]):
                mark_longer_edge(e, fullpath, truth_over, 'over')
            else:
                mark_shorter_edge(e, fullpath, truth_over, 'over')

        if len(truth_under)>0:
            if len(fullpath) >= len(truth_under[0]):
                mark_longer_edge(e, fullpath, truth_under, 'under')
            else:
                mark_shorter_edge(e, fullpath, truth_under, 'under')


def get_prev_markov_ps(curkpnet, pnets):
    kk = len(pnets)
    for e,d in curkpnet.edges.items():
        ee = e
        # for each of the lower orders
        for _k in range(kk,0,-1):
            currpkey = "pval-{}".format(_k)
            # init the log-prob
            mp = 0.0
            # get the corresponding path on the lower order
            ee = [v for v in honseq2prevseq(ee)]

            # for each lower-order edge on the path
            for st in range(len(ee)-1):
                # add to the log-probability
                try:
                    mp += pnets[_k].edges[(ee[st], ee[st+1])][currpkey]
                except KeyError:
                    mp += pnets[_k].edges[(ee[st], ee[st+1])]['pval']

            # assign the lower-order probability
            curkpnet.edges[e][currpkey] = mp

def get_pnets(paths, maxk, truth_over=[], truth_under=[]):
    pnets = {}
    #pickle.dump(paths, open('paths.pickle', 'wb'))
    #print('dumped paths.')
    #print(paths.paths[len(truth_over[0])])
    #print(truth_over)
    for k in range(1,maxk+1):
        hy_k = hypa.Hypa(paths, k=k)
        hy_k.construct_hypa_network(verbose=False)
        mark_truth(hy_k, truth_over=truth_over)
        if k > 1:
            get_prev_markov_ps(hy_k, pnets)
        pnets[k] = hy_k
    return pnets



################################################################################
################################################################################

def generate_pnets_with_anomaly(k_tru, num_seqs = 1500, anomaly_abundance = 0.3, maxk = 5, 
                                topo_dens = 0.05, nnodes1 = 50, nkedges = None):

    if nkedges is None:
        nkedges = int(nnodes1**2 * topo_dens / k_tru * 10)

    # Erdos-Renyi weighted topology
    net1 = pp.Network()
    for i in range(nnodes1):
        for j in range(nnodes1):
            # choose if there is a connection
            if random.random() <= topo_dens:
                # then draw a weight
                w = random.randint(1,20)
                net1.add_edge(i,j, weight=w)

    randpaths = pp.Paths()
    for _ in range(nkedges):
        randpaths.add_path(generate_walk(net1, k_tru))

    ## create the k-order network and add correlations
    hy_pnet = hypa.Hypa(randpaths, k=k_tru)
    hy_pnet.construct_hypa_network(verbose=False)

    truth_over = create_truth(hy_pnet, abundance=anomaly_abundance)

    paths_data = pp.Paths()
    avg_len = 0
    for _ in range(num_seqs):
        ww = pp.algorithms.random_walk.generate_walk(hy_pnet, l=10)
        walk_list = honwalk2firstwalk(ww)
        avg_len += len(walk_list)
        paths_data.add_path(walk_list)

    print("Average length of random walk: {}".format(avg_len/num_seqs))

    # Get the the HYPA networks for different orders
    pnets = get_pnets(paths_data, maxk=maxk, truth_over=truth_over)

    import pickle
    for k in pnets:
        with  open('edges-{}.pickle'.format(k), 'wb')  as pick:
            pickle.dump(pnets[k].edges, pick)
    #import sys
    #sys.exit()
    return pnets, truth_over, paths_data


################################################################################
################################################################################
################################################################################
def hypa_auc(max_k=3, n_samples=5):
    auroc = {kt:{k:[] for k in range(1,max_k+1)} for kt in range(2,max_k+1)}
    ## TODO I EDITED THIS RANGE, SHOULD START AT 2
    for kt in range(2, max_k+1):
        print("computing for implanted anomaly length={}...".format(kt))
        for _ in range(n_samples):
            pnets, _,_ = generate_pnets_with_anomaly(kt, maxk=max_k)
            auc_kt = compute_roc(pnets, plot=False, method='hypa')
            for k,val in auc_kt:
                auroc[kt][k].append(val)


    with open('output/randmod-auroc.pickle', 'wb') as f:
        pickle.dump(auroc, f)

    plt.figure()
    for kt,d in auroc.items():
        x = []
        y = []
        y_err = []
        for _x, vals in d.items():
            x.append(_x)
            y.append(np.nanmean(vals))
            y_err.append(np.nanstd(vals))

        y, y_err = np.array(y), np.array(y_err)

        plt.fill_between(x, y+y_err, y-y_err, alpha=0.25)
        plt.plot(x, y, 'o-', label='$l={}$'.format(kt))

    plt.plot((1,max(x)), (0.5,0.5), 'k--')
    plt.ylim((0.,1.05))
    plt.xticks(list(range(1, max(x)+1)), list(range(1, max(x)+1)))
    plt.xlabel('HYPA order')
    plt.ylabel('AUC')
    plt.legend(title='Anomaly length')
    plt.tight_layout()
    plt.savefig('output/randmod-auroc.pdf')


def fbad_auc(max_k=3, n_samples=5):
    '''
    Compute AUC for FBAD baseline.
    '''
    auroc = {kt:{k:[] for k in range(1,max_k+1)} for kt in range(2,max_k+1)}

    for kt in range(2, max_k+1):
        print("computing for implanted anomaly length={}...".format(kt))
        for _ in range(n_samples):
            pnets, _,_ = generate_pnets_with_anomaly(kt, maxk=max_k)
            auc_kt = compute_roc(pnets, plot=False, method='fbad', alpha=1.0)
            for k,val in auc_kt:
                auroc[kt][k].append(val)


    plt.figure()
    for kt,d in auroc.items():
        x = []
        y = []
        y_err = []
        for _x, vals in d.items():
            x.append(_x)
            y.append(np.nanmean(vals))
            y_err.append(np.nanstd(vals))

        y, y_err = np.array(y), np.array(y_err)
        plt.fill_between(x, y+y_err, y-y_err, alpha=0.25)
        plt.plot(x, y, 'o-', label='$l={}$'.format(kt))

    plt.plot((1,max(x)), (0.5,0.5), 'k--')
    plt.ylim((0.,1.05))
    plt.xticks(list(range(1, max(x)+1)), list(range(1, max(x)+1)))
    plt.xlabel('Detection order')
    plt.ylabel('AUC')
    plt.legend(title='Anomaly length')
    plt.tight_layout()
    plt.savefig('output/randmod-auroc-baseline.pdf')


if __name__=="__main__":
    import draw
    draw.set_style()
    plt.rcParams['figure.figsize'] =  [6.5, 6]
    plt.rcParams['axes.labelsize'] = 38
    plt.rcParams['xtick.labelsize'] = 26
    plt.rcParams['ytick.labelsize'] = 26
    plt.rcParams['legend.fontsize'] = 18

    print("Starting hypa_auc")
    hypa_auc(max_k=5, n_samples=5)
    #plt.clf()
    #print("Starting fbad_auc")
    #fbad_auc(max_k=5, n_samples=5)
