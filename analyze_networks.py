import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy
import math
import networkx as nx
from scipy.stats import norm, spearmanr
import os
import pandas as pd
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import operator
import matplotlib
import statsmodels.tsa.stattools

def get_centrality(G):
    node_centrality = collections.defaultdict(int)
    total = 0
    # Calculate the weighted edge centrality
    for node in G.nodes:
        for edge in G[node]:
            node_centrality[node] += abs(G[node][edge]['weight'])
            total += abs(G[node][edge]['weight'])

    # Normalise so the total is 1
    for comp in node_centrality:
        node_centrality[comp] = node_centrality[comp]/total

    sorted_centrality = sort_dict(node_centrality)
    centrality_names = [x[0] for x in sorted_centrality]
    centrality_sectors = []

    for name in centrality_names:
        centrality_sectors.append(G.nodes[name]['sector'])

    # Figure out the mean centrality of a sector
    sector_centrality = collections.defaultdict(float)
    no_companies_in_sector = collections.defaultdict(int)

    for comp in G:
        sector = G.nodes[comp]['sector']
        sector_centrality[sector] += node_centrality[comp]
        no_companies_in_sector[sector] += 1

    return node_centrality, sector_centrality

def turn_dict_into_np_array(dct, company_names):
    """
    Turns the dct into a numpy array where the keys are held in company_names
    """
    company_names = list(company_names)
    ret_arr = np.zeros(len(company_names))
    for key in dct:
        i = company_names.index(key)
        ret_arr[i] = dct[key]

    return ret_arr

def count_sector_connections(G):
    """
    Counts the number of connections from one sector to the next
    """
    sector_connections = collections.defaultdict(lambda: collections.defaultdict(int))
    for node in G:
        edges = G[node]
        sec1 = G.nodes[node]['sector']
        for edge in edges:
            sec2 = G.nodes[edge]['sector']
            sector_connections[sec1][sec2] += 1

    return sector_connections

def sort_dict(dct):
    """
    Takes a dict and returns a sorted list of key value pairs
    """
    sorted_x = sorted(dct.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_x

def save_open_figures(prefix=""):
    """
    Saves all open figures
    """
    figures=[manager.canvas.figure
         for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]

    for i, figure in enumerate(figures):
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        figure.savefig(prefix+'figure%d.png' % i)

def threshold_matrix(M, threshold):
    """
    Turns values below threshold to 0 and above threshold to 1
    """
    A = M.copy()
    low_value_indices = np.abs(A) < threshold
    A[low_value_indices] = 0
    high_value_indices = np.abs(A) > threshold
    A[high_value_indices] = 1
    return A

def get_sector_full_nice_name(sector):
    """
    Returns a short version of the sector name
    """       
    if sector == "information_technology":
        return "Information Technology"
    elif sector == "real_estate":
        return "Real Estate"
    elif sector == "materials":
        return "Materials"
    elif sector == "telecommunication_services":
        return "Telecommunication Services"
    elif sector == "energy":
        return "Energy"
    elif sector == "financials":
        return "Financials"
    elif sector == "utilities":
        return "Utilities"
    elif sector == "industrials":
        return "Industrials"
    elif sector == "consumer_discretionary":
        return "Consumer Discretionary"
    elif sector == "health_care":
        return "Healthcare"
    elif sector == "consumer_staples":
        return "Consumer Staples"
    else:
        raise Exception("%s is not a valid sector" % sector)

def plot_bar_chart(vals, label=None, title=None, xlabel=None, ylabel=None):
    fig = plt.figure()
    n = vals.shape[0]
    index = np.arange(n)
    bar_width = 0.1
    rects1 = plt.bar(index, vals, bar_width, label=label)
    #axes = fig.axes
    #print(axes)
    #axes[0].set_xticklabels(label)
    plt.xticks(index, label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

#df = pd.DataFrame.from_csv("s_and_p_500_sector_tagged.csv")
df = pd.read_csv("s_and_p_500_daily_close_filtered.csv", index_col=0)
company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
num_sectors = len(sectors)
company_sector_lookup = {}

for i,comp in enumerate(company_names):
    company_sector_lookup[comp] = company_sectors[i]

df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

window_size = 300
slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
dates = []

for x in range(no_runs-1):
    dates.append(df.index[(x+1)*slide_size+window_size][0:10])

dates_2 = []

for x in range(no_runs):
    dates_2.append(df.index[(x+1)*slide_size+window_size][0:10])

networks_folder = "networks_bic/"
onlyfiles = [os.path.abspath(os.path.join(networks_folder, f)) for f in os.listdir(networks_folder) if os.path.isfile(os.path.join(networks_folder, f))]
#onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
Graphs = []

# Sort the files into order
ind = [int(Path(x).stem[18:]) for x in onlyfiles]
ind = np.argsort(np.array(ind))

for i in ind:
    f = onlyfiles[i]
    G = nx.read_graphml(f)
    Graphs.append(G)

number_graphs = len(Graphs)
number_companies = len(G)
number_edges = []

clus_coef = []
average_degree = []
diameter = []
no_isolates = []
lcc = []
av_path_len = []
no_components = []
sector_centrality_lst = []
node_centrality_lst = []
sector_connections_lst = []
prec_fro_diff_lst = []
prev_prec = np.zeros((p, p))
prec_threshold_lst = []
prec_edge_diff = np.zeros(number_graphs)
prec_lst = []
sharpe_ratios = np.zeros(number_graphs*number_companies)
centralities = np.zeros(number_companies*number_graphs)
risks = np.zeros(number_companies*number_graphs)
ls = []

sharpe_correlations = []
sharpe_correlations_pvalues = []
risk_correlations = []
risk_correlations_pvalues = []
ret_correlations = []
ret_correlations_pvalues = []

for i,G in enumerate(Graphs):
    ls.append(G.graph['l'])
    prec = np.array(nx.to_numpy_matrix(G))
    number_edges.append(len(G.edges()))
    node_centrality, sector_centrality = get_centrality(G)
    sector_connections = count_sector_connections(G)
    sector_centrality_lst.append(sector_centrality)
    node_centrality_lst.append(node_centrality)
    no_isolates.append(len(list(nx.isolates(G))))
    sector_connections_lst.append(sector_connections)
    prec_lst.append(np.abs(prec))
    prec_thresh = threshold_matrix(prec, 0.0001)
    diff = np.count_nonzero(prec_thresh - prev_prec)
    prec_edge_diff[i] = diff
    prev_prec = prec_thresh
    prec_threshold_lst.append(prec_thresh)

    # Look at the returns
    if i + 1 < number_graphs:
        X_new = X[(i+1)*slide_size:(i+2)*slide_size+window_size, :]
        ret = np.sum(X_new, axis=0)
        risk = np.std(X_new, axis=0)

        sharpe = np.divide(ret, risk)
        centrality = turn_dict_into_np_array(node_centrality, company_names)
        sharpe_ratios[i*number_companies:(i+1)*number_companies] = sharpe.flatten()
        centralities[i*number_companies:(i+1)*number_companies] = centrality
        risks[i*number_companies:(i+1)*number_companies] = risk.flatten()

        corr, pvalue = spearmanr(centrality, sharpe)
        sharpe_correlations_pvalues.append(pvalue)
        sharpe_correlations.append(corr)
        corr, pvalue = spearmanr(centrality, risk)
        risk_correlations.append(corr)
        risk_correlations_pvalues.append(pvalue)
        corr, pvalue = spearmanr(centrality, ret)
        ret_correlations.append(corr)
        ret_correlations_pvalues.append(pvalue)
plt.figure()
plt.scatter(centralities, sharpe_ratios)
plt.title("Centrality Against Sharpe Ratio")
plt.xlabel("Centrality")
plt.ylabel("Sharpe Ratio")
print("Correlation between centrality and Sharpe Ratio:")
print(spearmanr(centralities, sharpe_ratios))

plt.figure()
plt.scatter(centralities, risks)
plt.title("Centrality Against Risk")
plt.xlabel("Centrality")
plt.ylabel("Risk")
print("Correlation between centrality and risks")
print(spearmanr(centralities, risks))

sector_connections_matrix = np.zeros((num_sectors, num_sectors))

number_of_sector_connections = np.zeros(num_sectors)

#total_sector_connections = collections.defaultdict(lambda: collections.defaultdict(int))
for sec_dict in sector_connections_lst:
    for sec1 in sec_dict:
        for sec2 in sec_dict[sec1]:
            i = sectors.index(sec1)
            j = sectors.index(sec2)
            sector_connections_matrix[i, j] += sec_dict[sec1][sec2]
            number_of_sector_connections[i] += 1

sector_nice_names = [get_sector_full_nice_name(x) for x in sectors]

plt.figure()
#matplotlib.rc('xtick', labelsize=16) 
#matplotlib.rc('ytick', labelsize=16) 
ax = sns.heatmap(sector_connections_matrix, cmap='Greys_r', xticklabels=sector_nice_names, yticklabels=sector_nice_names)
plt.title("Sum of sector connections")
plt.xticks(rotation=90)

number_edges = np.array(number_edges)
edges_changes = np.diff(number_edges)
prec_edge_diff = prec_edge_diff[1:]

dt = pd.to_datetime(dates_2)
ts = pd.Series(number_edges, index=dt)
fig = plt.figure()
ts.plot()
plt.title("Number of edges")

#What is the difference between these?
dt_2 = pd.to_datetime(dates)
ts = pd.Series(edges_changes, index=dt_2)
fig = plt.figure()
ts.plot()
plt.title("Changes in number of edges")

ts = pd.Series(prec_edge_diff, index=dt_2)
plt.figure()
ts.plot()
plt.title("Number of edge changes")

ts = pd.Series(sharpe_correlations, index=dt_2)
plt.figure()
ts.plot()
plt.title("Correlation between centrality and out of sample Sharpe ratio")

ts = pd.Series(sharpe_correlations_pvalues, index=dt_2)
plt.figure()
ts.plot()
plt.title("Correlation between centrality and out of sample Sharpe ratio pvalues")

ts = pd.Series(risk_correlations, index=dt_2)
ts.plot()
plt.title("Correlation between centrality and out of sample risk")

ts = pd.Series(ret_correlations, index=dt_2)
ts.plot()
plt.title("Correlation between centrality and out of sample return")

sector_centrality_over_time = collections.defaultdict(list)

for centrality in sector_centrality_lst: 
    s = sum(centrality.values())
    for sector in centrality:
        sector_centrality_over_time[sector].append(centrality[sector]/s)

sector_centrality = pd.DataFrame()
for sector in sector_centrality_over_time:
    ts = pd.Series(sector_centrality_over_time[sector], index=dt)
    sector_nice_name = get_sector_full_nice_name(sector)
    sector_centrality[sector_nice_name] = ts

sector_centrality.plot(color = ['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896'])

save_open_figures("financial_networks_graphml_")