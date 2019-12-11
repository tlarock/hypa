# hypa

`hypa` is an open source python package implementing Higher-Order Hypergeometric Path Anomaly Detection (HYPA), described in the following paper: 

Larock, T., Nanumyan, V., Scholtes, I., Casiraghi, G., Eliassi-Rad, T., Schweitzer, F. (2019) [Detecting Path Anomalies in Time Series Data on Networks](https://arxiv.org/abs/1905.10580). arXiv Prepr. arXiv1905.10580


## Requirements
`hypa` is written for python 3 and above and requires `pathpy`, a python package for analyzing sequential data using higher-order network models. You can install `pathpy` via `pip` using the command `pip install pathpy2`.

`hypa` also requires the [`rpy2` package](https://rpy2.readthedocs.io/en/version_2.8.x/getting-started.html). This package is used to interface with the R `ghypernet` package, available here: https://github.com/gi0na/r-ghypernet. Both need to be installed to work with `hypa`, and installation instructions for each package are available in their respective documentations.
