# hypa

`hypa` is an open source python package implementing Higher-Order Hypergeometric Path Anomaly Detection, described in the following paper: 

Larock, T., Nanumyan, V., Scholtes, I., Casiraghi, G., Eliassi-Rad, T., Schweitzer, F. (2019) [Detecting Path Anomalies in Time Series Data on Networks](https://arxiv.org/abs/1905.10580). arXiv Prepr. arXiv1905.10580


## Requirements
`hypa` is written for python 3+ and requires [`pathpy`](https://github.com/uzhdag/pathpy/tree/master/pathpy), a python package for analyzing sequential data using higher-order network models. You can install `pathpy` via `pip` using the command `pip install pathpy2`.

`hypa` also requires the [`rpy2` package](https://rpy2.readthedocs.io/en/version_2.8.x/getting-started.html). This package is used to interface with the R programming language, specifically the `ghypernet` package (available here: https://github.com/gi0na/r-ghypernet). Both need to be installed to work with `hypa`, and installation instructions for each package are available in their respective documentations.

## Installation
You can install `hypa` as follows:

1. Install and test the above requirements. *Important:* Do not proceed before starting a python session and ensuring that you can succesfully `import rpy2`.
2. Clone this repository using `git clone https://github.com/tlarock/hypa.git` from a terminal session. This command will download this repository to your computer in the current directory. Where you clone does not matter in principle, but it is best to store it somewhere where it will be safe from being deleted and where you can easily find it again later (e.g. your default `~/Downloads` directory may not be the best place).
3. Enter the cloned repository (`cd hypa`) and run the following command, which will install the package locally: `pip install . -e`
4. Test the package by starting a python session in a different directory (e.g. `cd ~`) and typing `import hypa`. 

If you have installation issues that you believe are directly related to `hypa`, please feel free to open an issue on this github repository. We do not maintain `rpy2` and so are probably not able to help with installation issues.

## Test
A simple test (based on the toy example in the paper) to make sure things are working is to paste the following code block into an `ipython` session:
```
import numpy as np
import pathpy as pp
import hypa
paths = pp.Paths()
paths.add_path(('A','X','C'), frequency=30)
paths.add_path(('B','X','D'), frequency=30)
paths.add_path(('B','X','C'), frequency=105)
print(paths)
hy = hypa.Hypa(paths)
hy.construct_hypa_network(k=2)
print(hy.hypa_net)
print(hy.hypa_net.edges)
for edge, edge_data in hy.hypa_net.edges.items(): 
  print("edge: {} hypa score: {}".format(edge, np.exp(edge_data['pval']))) 
```
