# hypa

`hypa` is an open source `python` package implementing Higher-Order Hypergeometric Path Anomaly Detection (HYPA) described in the following paper: Larock, T., Nanumyan, V., Scholtes, I., Casiraghi, G., Eliassi-Rad, T., Schweitzer, F. (2019) [Detecting Path Anomalies in Time Series Data on Networks](https://arxiv.org/abs/1905.10580). arXiv Prepr. arXiv1905.10580

## Quick Start Information
`hypa` uses the [`rpy2` package](https://rpy2.readthedocs.io/en/version_2.8.x/getting-started.html) to interface with the R `ghypernet` package, available in [this github repository](https://github.com/gi0na/r-ghypernet). Please follow the installation instructions for these packages. 

`hypa` also relies on  `pathpy`, a `python` package for dealing with higher-order sequential data. You can install `pathpy` via `pip` using the command `pip install pathpy2`. 
