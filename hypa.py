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

