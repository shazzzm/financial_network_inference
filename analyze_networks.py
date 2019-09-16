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
from sklearn.preprocessing import StandardScaler
import modularity_maximizer
from statsmodels.stats import multitest

def find_node_furthest_away(G, nodeset):
    """
    Finds the node that is furthest away in the graph from
    the nodes in the set
    """
    max_dist = -np.inf
    max_dist_node = None

    for node in G:
        if node in nodeset:
            continue
        # Find the total distance from the nodes in nodeset to node
        dist = 0
        for node2 in nodeset:
            dist += G[node][node2]['weight']

        if dist > max_dist:
            max_dist = dist
            max_dist_node = node

    return max_dist_node
 
def find_node_closest(G, nodeset):
    """
    Finds the node that is closest in the graph to
    the nodes in the set
    """
    min_dist = np.inf
    min_dist_node = None

    for node in G:
        if node in nodeset:
            continue
        # Find the total distance from the nodes in nodeset to node
        dist = 0
        for node2 in nodeset:
            dist += G[node][node2]['weight']

        if dist < min_dist:
            min_dist = dist
            min_dist_node = node

    return min_dist_node

def pick_portfolio_far_away(G, portfolio_size=5):
    """
    Picks portfolios of stocks that are far away from each other 
    using a greedy algorithm
    """
    new_G = remap_G_to_distance(G)
    portfolios = []
    for node in new_G.nodes():
        portfolio = set([node])

        while len(portfolio) < portfolio_size:
            portfolio.add(find_node_furthest_away(new_G, portfolio))

        portfolios.append(portfolio)

    return portfolios

def pick_portfolio_close(G, portfolio_size=5):
    """
    Picks portfolios of stocks that are close together using a 
    greedy algorithm
    """
    new_G = remap_G_to_distance(G)
    portfolios = []
    for node in new_G.nodes():
        portfolio = set([node])

        while len(portfolio) < portfolio_size:
            portfolio.add(find_node_closest(new_G, portfolio))

        portfolios.append(portfolio)

    return portfolios

def remap_G_to_distance(G):
    """
    Returns a new graph of edge distances from a correlation/partial correlation
    graph
    """
    new_G = G.copy()

    for edge in G.edges():
        weight = G.edges()[edge]['weight']
        new_weight = np.sqrt(2 * (1 - weight))
        new_G.edges[edge]['weight'] = new_weight

    return new_G

def select_portfolios_by_sharpe_ratio(portfolios, X, num_to_select=5):
    """
    Selects a set of portfolios that have the highest sharpe ratio on the current window
    """
    portfolio_sharpes = np.zeros(len(portfolios))
    for i,portfolio in enumerate(portfolios):
        portfolio = np.array(list(portfolio))
        num_stocks = portfolio.shape[0]
        weights = np.ones(num_stocks)/num_stocks
        ind = np.zeros(portfolio.shape, dtype=int)
        for k in range(portfolio.shape[0]):
            ind[k] = np.where(company_names == portfolio[k])[0]
        portfolio_returns = X[:, ind] @ weights
        mean_return = np.mean(portfolio_returns)
        stdev_return = np.std(portfolio_returns)

        portfolio_sharpes[i] = mean_return/stdev_return

    #ind = np.argsort(portfolio_sharpes)
    return portfolio_sharpes


def get_centrality(G):
    """
    Calculates the centrality of each node and mean centrality of a sector by
    summing the absolute values of each
    """
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

def min_max_normalization(df):
    return (df - df.min())/(df.max() - df.min())

def find_far_away_nodes(G):
    """
    Finds nodes that are anticorrelated with each other
    """
    pass

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

networks_folder = "networks_corr_lw/"
onlyfiles = [os.path.abspath(os.path.join(networks_folder, f)) for f in os.listdir(networks_folder) if os.path.isfile(os.path.join(networks_folder, f))]
#onlyfiles = onlyfiles[0:1]
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
prev_weighted_prec = np.zeros((p, p))
sharpe_ratios = np.zeros(number_graphs*number_companies)
centralities = np.zeros(number_companies*number_graphs)
risks = np.zeros(number_companies*number_graphs)
ls = []
edge_weights = []

sharpe_correlations = []
sharpe_correlations_pvalues = []
risk_correlations = []
risk_correlations_pvalues = []
ret_correlations = []
ret_correlations_pvalues = []

max_eigs = np.zeros(no_runs)

far_away_portfolio_sharpes = np.zeros(0)
close_portfolio_sharpes = np.zeros(0)
naive_portfolio_sharpes = []

mean_absolute_value = np.zeros(no_runs)

for i,G in enumerate(Graphs):
    ls.append(G.graph['l'])
    prec = np.array(nx.to_numpy_matrix(G))
    eigs = scipy.linalg.eigh(prec, eigvals=(p-1, p-1))[0][0]
    max_eigs[i] = eigs
    fro_diff = ((prec.flatten() - prev_weighted_prec.flatten())**2).mean()

    prec_fro_diff_lst.append(fro_diff)
    prev_weighted_prec = prec.copy()
    node_centrality, sector_centrality = get_centrality(G)
    sector_connections = count_sector_connections(G)
    sector_centrality_lst.append(sector_centrality)
    node_centrality_lst.append(node_centrality)
    no_isolates.append(len(list(nx.isolates(G))))
    sector_connections_lst.append(sector_connections)
    prec_lst.append(np.abs(prec))

    mean_absolute_value[i] = np.abs(prec).mean()
    edge_weights.append(prec.flatten())

    #far_away_portfolio_nodes = pick_portfolio_far_away(G)
    #close_portfolio_nodes = pick_portfolio_close(G)

    # Assess how the portfolios perform in this window to make a selection
    #far_away_portfolio_sharpes = np.concatenate((far_away_portfolio_sharpes, select_portfolios_by_sharpe_ratio(far_away_portfolio_nodes, X)))
    #close_portfolio_sharpes = np.concatenate((close_portfolio_sharpes, select_portfolios_by_sharpe_ratio(close_portfolio_nodes, X)))

    # Look at the returns
    if i + 1 < number_graphs:
        X_new = X[(i+1)*slide_size:(i+2)*slide_size+window_size, :]
        ret = np.mean(X_new, axis=0)
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

        weight_naive = np.ones(p)/p
        ret_naive = X_new @ weight_naive
        risk_naive = np.std(ret_naive)
        ret_naive = np.mean(ret_naive)

        naive_portfolio_sharpes.append(ret_naive/risk_naive)
        """
        for portfolio in far_away_portfolio_nodes:
            portfolio = np.array(list(portfolio))
            num_stocks = portfolio.shape[0]
            weights = np.ones(num_stocks)/num_stocks
            ind = np.zeros(portfolio.shape, dtype=int)
            for k in range(portfolio.shape[0]):
                ind[k] = np.where(company_names == portfolio[k])[0]
            portfolio_returns = X_new[:, ind] @ weights
            mean_return = np.mean(portfolio_returns)
            stdev_return = np.std(portfolio_returns)

            far_away_portfolio_sharpes.append(mean_return/stdev_return)


        for portfolio in close_portfolio_nodes:
            portfolio = np.array(list(portfolio))
            num_stocks = portfolio.shape[0]
            weights = np.ones(num_stocks)/num_stocks
            ind = np.zeros(portfolio.shape, dtype=int)
            for k in range(portfolio.shape[0]):
                ind[k] = np.where(company_names == portfolio[k])[0]
            portfolio_returns = X_new[:, ind] @ weights
            mean_return = np.mean(portfolio_returns)
            stdev_return = np.std(portfolio_returns)

            close_portfolio_sharpes.append(mean_return/stdev_return)
        """
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


dt = pd.to_datetime(dates_2)
dt_2 = pd.to_datetime(dates)

ts = pd.Series(mean_absolute_value, index=dt)
fig = plt.figure()
ts.plot()
plt.title("Mean Absolute Value")


ts = pd.Series(sharpe_correlations, index=dt_2)
plt.figure()
ax1 = ts.plot()
plt.title("Correlation between centrality and out of sample Sharpe ratio")
ax1.set_ylim(-0.5, 0.5)

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

ts = pd.Series(prec_fro_diff_lst[1:], index=dt_2)
plt.figure()
ts.plot()
plt.title("Mean Squared Edge Difference")

ts = pd.Series(max_eigs, index=dt)
plt.figure()
ts.plot()
plt.title("Largest Eigenvalue")

plt.figure()
plt.hist(edge_weights)
plt.title("Edge Weight Distribution")
ax = plt.gca()
ax.set_ylim(0, 85000)

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

macro_df = pd.read_csv('macroeconomic_variables.csv', index_col=0)
macro_df.index = pd.to_datetime(macro_df.index)
#sector_centrality_aligned = sector_centrality.resample('1M', convention='start').asfreq().dropna()  

sector_centrality_aligned = sector_centrality.to_period('M')
macro_df = macro_df.to_period('M')

# Pandas won't behave and calculate the correlation by itself, so we have to do it manually
corr_dct = collections.defaultdict(dict)
p_dct = collections.defaultdict(dict)
for var in macro_df:
    vals = macro_df[var]
    for sector in sector_centrality_aligned:
        vals2 = sector_centrality_aligned[sector].dropna()
        new_df = pd.DataFrame()
        new_df[var] = vals
        new_df[sector] = vals2
        new_df = new_df.dropna()
        Y = new_df.values
        corr, p = spearmanr(Y[:, 0], Y[:, 1])
        corr_dct[var][sector] = corr    
        p_dct[var][sector] = p     
        
corr_df = pd.DataFrame()
p_df = pd.DataFrame()
for key in corr_dct:
    corr_df[key] = pd.Series(corr_dct[key])

for key in p_dct:
    p_df[key] = pd.Series(p_dct[key])

print(multitest.multipletests(p_df, method='bonferroni'))

corr_df[p_df > 0.05] = 0  

ax = sns.heatmap(
    corr_df, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

plt.figure()
plt.boxplot([close_portfolio_sharpes, far_away_portfolio_sharpes, naive_portfolio_sharpes], labels=['Close Portfolios', 'Far Away Portfolios', 'Naive Sharpes'])

save_open_figures("financial_networks_graphml_")
plt.close('all')
