from scipy.stats import hypergeom

try:
    #from julia.api import Julia
    #jl = Julia(compiled_modules=False, runtime="/path/to/julia/julia")
    from julia.Distributions import Hypergeometric, cdf, logcdf, rand
except Exception as ioe:
    pass

try:
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
except ImportError as ioe:
    pass


class Hypa:
    '''
    Class for computing hypa scores on a DeBruijn graph given pathway data.
    '''
    def __init__(self, implementation):
        """
        Initialize Hypa object.
        """
        assert implementation in ['julia', 'rpy2', 'scipy'], "Invalid implementation."

        self.implementation = implementation

        # only import the relevant distribution function to be used in compute_hypa
        if self.implementation == 'julia':
            self.Hypergeometric = Hypergeometric
            self.cdf = cdf
            self.logcdf = logcdf
            self.rand = rand
        elif self.implementation == 'rpy2':
            ## import ghypernet from R
            rpy2.robjects.numpy2ri.activate()
            self.rphyper = ro.r['phyper']
            self.rrhyper = ro.r['rhyper']
        elif self.implementation == 'scipy':
            # Importing scipy in hypa_* files.
            pass
