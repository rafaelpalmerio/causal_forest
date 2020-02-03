# causal_forest

Python implementation of the following paper: 
https://arxiv.org/pdf/1510.04342.pdf

Usage:
from sklearn.datasets import make_regression
n_samples = 2000
X, y = make_regression(n_samples=n_samples)
X = pd.DataFrame(X)
X['target'] = y
X['w'] = 0
X.loc[n_samples // 2:, 'target'] = X.loc[n_samples // 2:, 'target']*2
X.loc[n_samples // 2:, 'w'] = 1


a = CausalForest()
X = X.reset_index().rename(columns={'index':'index1'})
a.fit(X, 'target', 'w', index_cols=['index1'], min_samples_leaf=10, n_estimators=40, algorithm='propensity',
      n_threads=1, full_predictor=True).mean(axis=1).mean()
      
a.plot_results(X)
