"""
Precision matrix can be used for predictions, so we see how they compare as time goes on
"""
import numpy as np
import scipy
import pandas as pd
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

def predict_from_precision_matrix(X, prec, i):
    """
    Predicts the value of X_i usig the rest of X and the precision matrix
    """
    n, p = X.shape
    indices = np.arange(p)
    X_i = X[:, indices!=i]
    beta = -prec[indices!=i, i] / prec[i, i]
    X_bar = X_i @ beta
    return X_bar

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
df = pd.read_csv("s_and_p_500_daily_close_filtered.csv", index_col=0)

company_sectors = df.iloc[0, :].values
company_names = df.T.index.values
sectors = list(sorted(set(company_sectors)))
df_2 = df.iloc[1:, :]
df_2 = df_2.apply(pd.to_numeric)
df_2 = np.log(df_2) - np.log(df_2.shift(1))
X = df_2.values[1:, :]

window_size = 300
slide_size = 30
no_samples = X.shape[0]
p = X.shape[1]
no_runs = math.floor((no_samples - window_size)/ (slide_size))
print("We're running %s times" % no_runs)

dates = df.index[2:]

matrices_folder = "precision_matrices_lw/"
onlyfiles = [os.path.abspath(os.path.join(matrices_folder, f)) for f in os.listdir(matrices_folder) if os.path.isfile(os.path.join(matrices_folder, f))]
#onlyfiles = list(map(lambda x: os.path.splitext(x)[0], onlyfiles))
matrices = []
# Sort the files into order
ind = [int(Path(x).stem[5:]) for x in onlyfiles]
ind = np.argsort(np.array(ind))

for i in ind:
    f = onlyfiles[i]
    m = np.load(f)
    matrices.append(m)

dt = pd.to_datetime(dates)

for i,G in enumerate(matrices):
    X_new = X[i*slide_size:, :]
    X_new = X_new - X_new.mean(0)
    new_n = X_new.shape[0]
    prec = matrices[i]
    res_norms = np.zeros(p)
    reses = np.zeros((new_n, p))
    for j in range(p):
        X_hat = predict_from_precision_matrix(X_new, prec, j)
        res = np.divide(X_new[:, j] - X_hat, X_new[:, j])
        res_norms[j] = np.linalg.norm(res)
        reses[:, j] = res

    # Get the largest residual
    ind = np.argsort(res_norms)[::-1]
    max_res = ind[0]
    min_res = ind[-1]


    ts_min = pd.Series(reses[:, min_res], index=dt[i*slide_size:])
    ts_max = pd.Series(reses[:, max_res], index=dt[i*slide_size:])

    print("Min error is %s, %s" % (company_names[min_res], company_sectors[min_res]))
    print("Max error is %s, %s" % (company_names[max_res], company_sectors[max_res]))

    fig = plt.figure()
    ts_min.plot()
    plt.title("Residual for company %s at %s (min)" % (company_names[min_res], i))
    plt.savefig("res_%s_%s" % (i, min_res))
    plt.close()

    fig = plt.figure()
    ts_max.plot()
    plt.title("Residual for company %s at %s (max)" % (company_names[max_res], i))
    plt.savefig("res_%s_%s" % (i, max_res))
    plt.close()
    #values = pd.DataFrame()

    #values["Predicted"] = pd.Series(X_hat, index=dt[i*slide_size:])
    #values["Actual"] = pd.Series(X_new[:, 0], index=dt[i*slide_size:])

    #fig = plt.figure()
    #values.plot()
    #plt.title("Predicted vs Actual for company %s at %s" % (j, i))
    #plt.savefig("pred_%s_%s" % (i, j))
    #plt.close()

 
#save_open_figures("financial_networks_graphml_")
plt.close('all')