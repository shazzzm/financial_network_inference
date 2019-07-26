import pandas as pd
import numpy as np
from sklearn.covariance import GraphLassoCV
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import network_analysis as nta
import collections
import scipy
import math
import networkx as nx
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import space


#df = pd.DataFrame.from_csv("s_and_p_500_sector_tagged.csv")
#df = pd.DataFrame.from_csv("s_and_p_500_daily_close_filtered.csv")
df = pd.read_csv("s_and_p_500_daily_close_filtered.csv", index_col=0)


company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]
#X = df_2.pct_change(axis=0).values[1:, :]

window_size = 300
slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))

print("We're running %s times" % no_runs)



X_new = X[0:window_size, :]
ss = StandardScaler()
X_new = ss.fit_transform(X_new)
s = space.SPACE_BIC(verbose=True)
s.fit(X_new)
prec = s.precision_
l = s.alpha_
np.fill_diagonal(prec, 0)

G=nx.from_numpy_matrix(prec)
G=nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
nx.set_node_attributes(G, node_attributes, 'sector')
G = nx.from_numpy_matrix(prec)
G.graph['l'] = l
nx.write_graphml(G, "network_over_time_%s.graphml" % 0)
print("%s non-zero values" % np.count_nonzero(prec))

prev_prec = prec.copy()

for x in range(1, no_runs):
    print("Run %s" % x)
    X_new = X[x*slide_size:(x+1)*slide_size+window_size, :]
    ss = StandardScaler()
    X_new = ss.fit_transform(X_new)
    s = space.SPACE_BIC(l2_reg=0.5, verbose=True)
    s.fit(X_new)
    prec = s.precision_
    l = s.alpha_
    np.fill_diagonal(prec, 0)
    print("%s non-zero values" % np.count_nonzero(prec))
    G=nx.from_numpy_matrix(prec)
    G=nx.relabel_nodes(G, dict(zip(G.nodes(), company_names)))
    node_attributes = dict(zip(company_names[list(range(len(company_sectors)))], company_sectors))
    nx.set_node_attributes(G, node_attributes, 'sector')
    G = nx.from_numpy_matrix(prec)
    G.graph['l'] = l
    nx.write_graphml(G, "network_over_time_%s.graphml" % x)