import sys
import os
import traceback
import subprocess


def convert_paths(paths):
    '''
    Accepts a paths object and converts the paths to
    SPMF formatted string. Saves mapping of SPMF indices to
    node names in paths object.
    '''
    idx = 0
    mapping = dict()
    out_str = ''
    for k, kpaths in paths.paths.items():
        for path in kpaths:
            for _ in range(int(kpaths[path][1])):
                for i in range(len(path)):
                    item = path[i]
                    if item not in mapping:
                        mapping[item] = idx
                        idx += 1
                    out_str += str(mapping[item])
                    out_str += ' -1 '
                out_str += '-2\n'

    return out_str, mapping

def run_promise(converted_paths_str, mapping, output_filename='tmp', p=50, t=500, theta=0.0, cores=2, strategy=1, promise_path = '../../PROMISE/', redirect_output=True):
    '''
    Accepts an SPMF formatted string, prints it to the promise/data directory
    at output_filename.txt, runs ProMiSe, reads output, returns significant paths.
    '''
    
    ## Write SPMF string to a file that ProMiSe.jar can find
    try:
        data_path = promise_path + 'data/' + output_filename + '.txt'
        f_out = open(data_path, 'w')
        f_out.write(converted_paths_str)
        f_out.close()
    except IOError as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
        sys.exit()

    ## Set up and execute ProMiSe.jar
    PROMISE_args = ['java' , '-Xmx12G', '-jar', 'ProMiSe.jar', output_filename, p, t, theta, cores, strategy]
    PROMISE_args = list(map(str, PROMISE_args))
    try:
        ## save current working directory
        cwd = os.getcwd()
        ## switch to ProMiSe directory and execute Java code
        os.chdir(promise_path)
        if not redirect_output:
            subprocess.run(PROMISE_args, check=True)
        else:
            subprocess.run(PROMISE_args, check=True, \
                           stdout=open("/dev/null", 'w'), stderr=open("/dev/null", 'w'))
        
        ## return to hypa working directory
        os.chdir(cwd)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
        sys.exit()

    ## Read output of ProMiSe.jar and convert back to original node names
    reverse_mapping = {str(val):key for key,val in mapping.items()}
    try:
        sfsp_file = promise_path + 'data/' + output_filename + '_SFSP.txt'
        anom_paths = []
        f_in = open(sfsp_file)
        for line in f_in:
            s = line.split(' ')
            path = []
            for entry in s:
                if entry == '-1':
                    continue
                elif entry == '#SUP:':
                    break
                else:
                    path.append(reverse_mapping[entry])

            anom_paths.append(path)
        f_in.close()
        os.remove(sfsp_file)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
        anom_paths=[]

    return anom_paths

def compute_promise(networks, paths_data, wy_datasets=50, mc_datasets=1024, \
            minimum_frequency=0.0001, cores=2, strategy=1, promise_path='../../PROMISE/', redirect_output=True, outfile='tmp'):
    '''
    Return anomalous paths according to PROMISE
    '''
    ## Convert paths object to SPMF format
    converted_paths_str, mapping = convert_paths(paths_data)
    ## Run ProMiSe.jar on paths
    anomalous_paths = run_promise(converted_paths_str, mapping, p=wy_datasets,\
                                  t=mc_datasets, theta=minimum_frequency, \
                                  cores=cores, promise_path=promise_path, \
                                  redirect_output=redirect_output, output_filename=outfile)

    ## Mark anomalous edges in networks object 
    for path in anomalous_paths:
        k = len(path) - 1
        if k in networks:
            edge = ','.join(path[0:k]), ','.join(path[1:k+1])
            networks[k].edges[edge]['promise'] = True

    return networks

if __name__ == '__main__':
    import hypa
    import pathpy as pp
    paths = pp.Paths()
    paths.add_path(['a','x','c'], frequency=30)
    paths.add_path(['b','x','c'], frequency=105)
    paths.add_path(['b','x','d'], frequency=100)

    pnets = dict()
    hy = hypa.Hypa(paths)
    for k in [1, 2]:
        hy.construct_hypa_network(k=k, verbose=False)
        pnets[k] =  hy.hypa_net

    pnets = compute_promise(pnets, paths, promise_path='/scratch/larock.t/PROMISE/')
