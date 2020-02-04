import random
import pickle
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pathpy as pp
from pathpy.algorithms.random_walk import generate_walk

import hypa

from sklearn import metrics

def create_truth(pnet, abundance=0.3, concentrate_all = True):
    truth_over = []
    for e, nei in pnet.successors.items():
        if len(nei) < 2 or random.random()<abundance:
            continue

        neiwei = [(t,pnet.edges[(e,t)]['weight']) for t in nei]
        neiwei = sorted(neiwei, key=lambda r: r[1])#.sum())

        # leave only one with all the weights
        if concentrate_all:
            tokeep = neiwei[-1]
            allwei = 0
            for t,w in neiwei[:-1]:
                allwei += w
                pnet.edges[(e,t)]['weight'] = 0
                ## to mark these under is probably useless, they were random anyway
                # pnet.edges[(e,t)]['truth'] = 'under'
            pnet.edges[(e,tokeep[0])]['weight'] += allwei
            pnet.edges[(e,tokeep[0])]['truth'] = 'over'

            truth_over.append(honwalk2firstwalk( (e,tokeep[0]) ))

        else:
            # make the max under-rep by giving all its weight to another
            # then set its weight to 0 and mark "under"
            # mark the other "over"
            tound = neiwei[-1]
            pnet.edges[(e,tound[0])]['weight'] = 0.#np.array([0.,0.])
            pnet.edges[(e,tound[0])]['truth'] = 'under'

            toov = neiwei[-2]
            pnet.edges[(e,toov[0])]['weight'] += tound[1]
            pnet.edges[(e,toov[0])]['truth'] = 'over'

            truth_over.append(honwalk2firstwalk( (e,toov[0]) ))

    return truth_over

def paths_from_hon(hon, sep=','):
    paths = pp.Paths()
    for e,d in hon.edges.items():
        freq = int(d['weight'])
        if freq == 0:
            continue

        pathe = sep.join([e[0], e[1].split(sep)[-1]])
        paths.add_path(pathe, frequency=freq)

    return paths


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


def compute_roc(pnets, truth_k=2, plot=True, output=None, method='hypa', alpha=0.5):
    """
    Compute Reciever Operating Characteristic for a given network
    """
    assert method in ['hypa', 'fbad', 'promise'], \
            "method must be one of hypa, fbad, or promise; not {}".format(method)

    k = max(list(pnets.keys()))
    auc_k = []

    for _k in range(1,k+1):
        if method == 'baseline':
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
            elif method == 'baseline':
                if d['weight'] > (mean + std*alpha):
                    y_score.append(1.0)
                else:
                    y_score.append(0.0)
            elif method == 'promise':
                if 'promise' in d.keys():
                    y_score.append(1.0)
                else:
                    y_score.append(0.0)

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
    for k in range(1,maxk+1):
        hy_k = hypa.Hypa(paths)
        hy_k.construct_hypa_network(k=k, verbose=False)
        pnet = hy_k.hypa_net
        mark_truth(pnet, truth_over=truth_over)
        if k > 1:
            get_prev_markov_ps(pnet, pnets)
        pnets[k] = pnet
    return pnets



################################################################################
################################################################################

def generate_random_path_data(net1, k_tru, num_seqs = 1500, anomaly_abundance = 0.3, maxk = 5, nnodes1=50,topo_dens=0.05, expand_subpaths=True):
    nkedges = int(nnodes1**2 * topo_dens / k_tru * 10)
    randpaths = pp.Paths()
    for _ in range(nkedges):
        randpaths.add_path(generate_walk(net1, k_tru), expand_subpaths=expand_subpaths)

    randpaths.expand_subpaths()
    ## create the k-order and add correlations
    hy_pnet = hypa.Hypa(randpaths)
    hy_pnet.construct_hypa_network(k=k_tru, verbose=False)
    pnetkcorr = hy_pnet.hypa_net

    truth_over = create_truth(pnetkcorr, abundance=anomaly_abundance)

    seq_data = []
    for _ in range(num_seqs):
        ww = pp.algorithms.random_walk.generate_walk(pnetkcorr, l=nnodes1)
        seq_data.append(honwalk2firstwalk(ww))

    paths_data = pp.Paths()
    for ww in seq_data:
        paths_data.add_path(ww, expand_subpaths=expand_subpaths)

    return truth_over, paths_data




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

    ## create the k-order and add correlations
    hy_pnet = hypa.Hypa(randpaths)
    hy_pnet.construct_hypa_network(k=k_tru, verbose=False)
    pnetkcorr = hy_pnet.hypa_net

    truth_over = create_truth(pnetkcorr, abundance=anomaly_abundance)

    seq_data = []
    for _ in range(num_seqs):
        #ww = pp.algorithms.random_walk.generate_walk(pnetkcorr, l=nnodes1)
        ww = pp.algorithms.random_walk.generate_walk(pnetkcorr, l=10)
        seq_data.append(honwalk2firstwalk(ww))

    paths_data = pp.Paths()
    for ww in seq_data:
        paths_data.add_path(ww)

    # Get the the HYPA networks for different orders
    pnets = get_pnets(paths_data, maxk=maxk, truth_over=truth_over)

    return pnets, truth_over, paths_data


################################################################################
################################################################################
################################################################################
def hypa_auc(max_k=3, n_samples=5):
    auroc = {kt:{k:[] for k in range(1,max_k+1)} for kt in range(2,max_k+1)}

    for kt in range(2, max_k+1):
        print("computing for implanted anomaly length={}...".format(kt))
        for _ in range(n_samples):
            pnets, _,_ = generate_pnets_with_anomaly(kt, maxk=max_k)
            auc_kt = compute_roc(pnets, kt, plot=False)
            for k,val in auc_kt:
                auroc[kt][k].append(val)


    with open('output/randmod-auroc.pickle', 'wb') as f:
        pickle.dump(auroc, f)

    draw.set_style()

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
            auc_kt = compute_roc(pnets, kt, plot=False, baseline=True, alpha=1.0)
            for k,val in auc_kt:
                auroc[kt][k].append(val)


    draw.set_style()
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
    plt.xlabel('Detection order')
    plt.ylabel('AUC')
    plt.legend(title='Anomaly length')
    plt.tight_layout()
    plt.savefig('output/randmod-auroc-baseline.pdf')


def PROMISE_auc(max_k=3, n_samples=5, wy_datasets=50, mc_datasets=1024, minimum_frequency=0.0001, cores=1, strategy=1, promise_path='../../PROMISE/',redirect_output=True, outfile='tmp'):
    '''
    Compute AUC for PROMISE baseline.
    '''
    from promise import compute_promise
    auroc = {kt:{k:[] for k in range(1,max_k+1)} for kt in range(2,max_k+1)}

    ## I need to make d['promise'] be True if edge is anomalous according to
    ## PROMISE, false otherwise
    for kt in range(3, max_k+1):
        print("computing for implanted anomaly length={}...".format(kt), flush=True)
        sample = 0
        while sample < n_samples:
            print("Sample: {}".format(sample), flush=True)
            pnets, _, paths_data = generate_pnets_with_anomaly(kt, maxk=max_k, num_seqs = 2000)
            pnets = compute_promise(pnets, paths_data, wy_datasets, mc_datasets, minimum_frequency, cores, strategy, promise_path, \
                                 outfile=outfile + '-{}-{}'.format(kt, sample), redirect_output=redirect_output)
            
            if not pnets:
                continue

            auc_kt = compute_roc(pnets, kt, plot=False, method='promise', alpha=1.0)
            for k,val in auc_kt:
                auroc[kt][k].append(val)

            with open('output/auroc-{}_theta-{}_P-{}_T-{}.pickle'.format(kt, minimum_frequency, wy_datasets, mc_datasets), 'wb') as f:
                pickle.dump(auroc, f)
            sample += 1

    draw.set_style()
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
    plt.xlabel('Detection order')
    plt.ylabel('AUC')
    plt.legend(title='Anomaly length')
    plt.tight_layout()
    plt.savefig('output/randmod-auroc-promise_theta-{}_P-{}_T-{}.pdf'.format(minimum_frequency, wy_datasets, mc_datasets))


if __name__=="__main__":
    import draw
    draw.set_style()

    #print("Startin hypa_auc")
    #hypa_auc(max_k=5, n_samples=10)
    #print("Starting fbad_auc")
    #fbad_auc(max_k=5, n_samples=10)
    minimum_frequency=0.1
    wy_datasets=1000
    mc_datasets=10048
    cores=64
    promise_path='../../PROMISE/'
    promise_path='/scratch/larock.t/PROMISE/'
    #PROMISE_auc(max_k=4, n_samples=5, wy_datasets=25, mc_datasets=150, cores=56, promise_path='/scratch/larock.t/PROMISE/')
    PROMISE_auc(max_k=5, n_samples=2, wy_datasets=wy_datasets, mc_datasets=mc_datasets, \
                promise_path=promise_path, minimum_frequency=minimum_frequency, cores=cores, redirect_output=False, \
                outfile='tmp-{}-{}-{}'.format(int(minimum_frequency*100), wy_datasets, mc_datasets))
