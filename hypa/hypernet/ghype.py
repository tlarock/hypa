import numpy as np

class ghype():
    def __init__(self, ghype_r):
        self.ghype_r = ghype_r
        self.adj = np.array(self.get_attr('adj'))
        self.xi = np.array(self.get_attr('xi'))
        self.m = self.get_attr('m')[0]

    def get_attr(self, attri):
        for key,val in self.ghype_r.items():
            if key == attri:
                return val

        return None

    def copy(self):
        return ghype(self.ghype_r)

