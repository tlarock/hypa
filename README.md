# hypa

`hypa` is an open source python package implementing Higher-Order Hypergeometric Path Anomaly Detection, described in the following paper: 

Larock, T., Nanumyan, V., Scholtes, I., Casiraghi, G., Eliassi-Rad, T., Schweitzer, F. (2019) [Detecting Path Anomalies in Time Series Data on Networks](https://arxiv.org/abs/1905.10580). arXiv Prepr. arXiv1905.10580

In the `reproduce/` directory of this repository you can find a Python file called `anomalous_random.py`, which can reproduce Figure 3. You can also find a Jupyter notebook called `flight_analysis.ipynb` that can reproduce Figure 4. 

## Requirements
`hypa` is written for python 3+ and requires [`pathpy`](https://github.com/uzhdag/pathpy/tree/master/pathpy), a python package for analyzing sequential data using higher-order network models. You can install `pathpy` via `pip` using the command `pip install pathpy2`.

The python implementation of the Hypergeometric distribution, [`scipy.stats.hypergeom`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html#scipy.stats.hypergeom), is not as precise as either the [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Hypergeometric) or [`R.stats`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Hypergeometric.html) versions, and the `hypergeom.logcdf` calculation, the most important for HYPA, is very slow. 

Due to this, I have made all 3 implementations accessible in this package. Note that once you have the implementation you want installed, all you need to do is pass the correct name to the `Hypa(paths, implementation='julia')` constructor. It should not be necessary to interface directly with the specific implementations. Details are as follows:

1. `implementation='julia'`: The Julia implementatioon in `Distributinos.jl` is the default because it is the fastest and slighty simpler to install and work with (in my experience) than `rpy2`. We use [PyJulia](https://pyjulia.readthedocs.io/en/latest/index.html) to access Julia from Python. You can follow the instructions there for the implementation, but in general you will need to have Julia installed on your machine along with `Distributions.jl` and [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl).

2. `implementation='rpy2'`: The `R.stats` implementation is as effective as (if slightly slower than) `Distributions.jl`, but installing [`rpy2`](https://rpy2.github.io/) to access R from Python tends to more finicky (in my experience). 

3. `implementation='scipy'`: Using `scipy.stats.hypergeom` is the simplest, but also the worst performing. Any `scipy.stats` distribution should have `hypergeom` and there should be no trouble with imports. We recommend *not* computing the CDF in log space (e.g. set `log=False` in `Hypa.construct_hypa_network`), since the `logcdf` function is very slow. For reasons still mysterious to me, the `cdf` function in this version seems to have some numerical issues that the others do not.

## Installation
You can install `hypa` as follows:

1. Install and test the above requirements.
2. Clone this repository using `git clone https://github.com/tlarock/hypa.git` from a terminal session. This command will download this repository to your computer in the current directory. Where you clone does not matter in principle, but it is best to store it somewhere where it will be safe from being deleted and where you can easily find it again later (e.g. your default `~/Downloads` directory may not be the best place).
3. Enter the cloned repository (`cd hypa`) and run the following command, which will install the package locally: `pip install -e .` (if you do not have root access where you are making the installation, try `pip install --user -e .` instead.) 
4. Test the package by starting a python session in a different directory (e.g. `cd ~`) and typing `import hypa`. 

If you have installation issues that you believe are directly related to `hypa`, please feel free to open an issue on this github repository. We do not maintain any of the other dependencies and so are probably not able to help with installation issues, thuogh you are free to ask.

## Test
A simple test (based on the toy example in the paper) to make sure things are working is to paste the following code block into an `ipython` session or run it as a script:
```
import numpy as np
import pathpy as pp
import hypa
paths = pp.Paths()
paths.add_path(('A','X','C'), frequency=30)
paths.add_path(('B','X','D'), frequency=100)
paths.add_path(('B','X','C'), frequency=105)
print(paths)
hy = hypa.HypaPP.from_paths(paths, k=2, implementation='julia') # Insert your desired implementation (out of 'julia', 'rpy2', 'scipy') here!
print(hy.hypa_net)
print(hy.hypa_net.edges)
for edge, edge_data in hy.hypa_net.edges.items(): 
  print("edge: {} hypa score: {}".format(edge, np.exp(edge_data['pval']))) 
```
