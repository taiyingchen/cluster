# GMM

Python implementation of Gaussian Mixture Model (GMM)\\
Currently only support data points in **2 dimensions**

## Installation

```bash
git clone https://github.com/taiyingchen/gmm.git
cd gmm/
pip3 install .
```

### Dependencies

Required only Python (>= 3.6)

## Usage

```python
from gmm import GMM

X = [[8.98320053625, -2.08946304844],
        [2.61615632899, 9.46426282022],
        [1.60822068547, 8.29785986996],
        [8.64957587261, -0.882595891607],
        [1.01364234605, 10.0300852081],
        [1.49172651098, 8.68816850944],
        [7.95531802235, -1.96381815529],
        [0.527763520075, 9.22731148332],
        [6.91660822453, -3.2344537134],
        [6.48286208351, -0.605353440895]]
# Initial centroid of cluster
clusters = [[3.35228193353, 6.27493570626], [6.76656276363, 6.54028732984]]

gmm = GMM(max_iter=100)
gmm.fit(X, clusters)
y = gmm.predict(X)
print(y)
# [1, 0, 0, 1, 0, 0, 1, 0, 1, 1]
```

## Todo

* Support data points in multi-dimensions
* Import `numpy` library for matrix calculation
