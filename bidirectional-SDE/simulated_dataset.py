import numpy as np
import scipy as sp
import scanpy as sc
import anndata as ad
import pandas as pd
import os
import matplotlib.pyplot as plt
import networkx as nx
PLT_CELL = 3.5
np.random.seed(0)
# set up simulation parameters
A_true = np.zeros((8, 8))
d = A_true.shape[0]
N = 100
sigma = np.eye(d)*0.05
for i in range(A_true.shape[0]):
    A_true[i, (i+1) % d] = 1
A_true[-1, 0] = -1
A_true = 1.25*(A_true - np.eye(d))
import sys
sys.path.append("../src/")
import util
import seaborn as sb
import importlib
importlib.reload(util)
T = 10
t1 = 10
dt = 0.01
ts = np.linspace(0, t1, T)
x0 = (np.arange(d) == 0) * 0.25
# knockouts
kos = []
Ms = [np.ones((d, d), ), ]
ko_label = ["wt", ]
for i in kos:
    M = np.ones((d, d))
    M[:, i] = 0
    Ms.append(M)
    ko_label.append(str(i))
kos = [None, ] + [str(i) for i in kos]
# simulate WT and all knockouts
xs = []
for M in Ms:
    xs.append(util.simulate(A_true * M, sigma, N, ts, lambda N, d: x0 + np.random.randn(N, d)*0.05))
xs = np.stack(xs)
plt.figure(figsize = (3.5, 3))
sb.heatmap(A_true, cmap = "RdBu_r")
plt.title("$A$")
import networkx as nx
_A = A_true.copy()
np.fill_diagonal(_A, 0)
g = nx.DiGraph(np.abs(_A))
centralities = nx.centrality.eigenvector_centrality(g)
plt.figure(figsize = (3, 3))
nx.draw(g, with_labels = True)
import sklearn as sk
from sklearn import decomposition
pca_op = sk.decomposition.PCA(n_components = d)
pca_op.fit(xs.reshape(-1, d))
ys = pca_op.transform(xs.reshape(-1, d)).reshape(xs.shape)
from matplotlib import cm, colors
norm = colors.Normalize(vmin = 0, vmax = T)
cmap = cm.viridis
m = cm.ScalarMappable(norm=norm, cmap=cmap)
plt.figure(figsize = (4/3*PLT_CELL, PLT_CELL))
for j in range(T):
    plt.scatter(ys[0, j, :, 0], ys[0, j, :, 1], alpha = 0.5, color = m.to_rgba(j))
cb = plt.colorbar()
cb.solids.set(alpha=1)
plt.xlabel("PCA1"); plt.ylabel("PCA2")
plt.tight_layout()
plt.savefig("../figures/8D_snapshots.pdf")