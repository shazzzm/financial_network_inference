import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import space_r
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.stats import norm
import sklearn.datasets

class SPACE():
    """
    Python implementation of SPACE. We use the C version of the Joint Sparse Regression model 
    """
    def __init__(self, l1_reg, l2_reg=0, sig=None, weight=None):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.l1_reg_ctype = ctypes.c_float(self.l1_reg)
        self.l2_reg_ctype = ctypes.c_float(self.l2_reg)

        self.sig = sig
        self.weight = weight

        self.lib = ctypes.CDLL("jsrm.so")   
        self.fun = self.lib.JSRM
        self.fun.restype = None
        self.fun.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), 
            ctypes.POINTER(ctypes.c_float), ndpointer(ctypes.c_float),ndpointer(ctypes.c_float), ctypes.POINTER(ctypes.c_int), 
            ctypes.POINTER(ctypes.c_int), ndpointer(ctypes.c_float)]
        self.precision_ = None


    def run_jsrm(self, X):
        X = X.copy()
        n, p = X.shape
        n_in = ctypes.c_int(n)
        p_in = ctypes.c_int(p)
        sigma_sr = self.sig.astype(np.float32)
        n_iter = ctypes.c_int(100)
        iter_count = ctypes.c_int(0)
        beta = np.zeros(p**2, dtype=np.float32)
        n_iter_out = ctypes.c_int(0)

        X = X.astype(np.float32)
        X_in = X.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        sigma_sr_in = sigma_sr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        beta_out = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        space_prec = np.reshape(beta, (p, p))
        self.fun(ctypes.byref(n_in), ctypes.byref(p_in), ctypes.byref(self.l1_reg_ctype), ctypes.byref(self.l2_reg_ctype), X, sigma_sr, ctypes.byref(n_iter), ctypes.byref(n_iter_out), beta)
        space_prec = np.reshape(beta, (p, p))

        return space_prec

    def fit(self, X ,iter=2):
        n, p = X.shape
        update_sig = False
        if self.sig is None:
            update_sig = True
            self.sig = np.ones(p, dtype=np.float32)

        update_weight = False
        if self.weight is None:
            update_weight = False
            self.weight = np.ones(p, dtype=np.float32)

        for i in range(iter):
            space_prec = self.run_jsrm(X)
            np.fill_diagonal(space_prec, 1)
            ind = np.triu_indices(p)
            coef = space_prec[ind]
            beta = self.Beta_coef(coef, self.sig)
        
        
            if not update_sig and not update_weight:
                break
            else: ## update sig or weight
                if update_sig:
                    self.sig = self.InvSig(X, beta)   
            ### end updating WEIGHT and SIG
        ### end iteration

        self.precision_ = space_prec

    def InvSig (self, X, beta):
        ################### parameters
        ### X:    n by p data matrix; should be standardized to sd 1;
        ### beta: beta matrix for the regression model
        n,p = X.shape
        beta = beta.copy()
        np.fill_diagonal(beta, 0)
        X_hat = X @ beta
        residue = X - X_hat
        result = (1/n) * np.power(residue, 2).sum(axis=0)
        return np.reciprocal(result)

    def Beta_coef(self, coef, sig):
        ############## parameter
        ### coef: rho^{ij}; 
        ### sig: sig^{ii}
        p = sig.shape[0]
        result = np.zeros((p, p), dtype=np.float32)
        ind = np.triu_indices(p)

        result[ind] = coef
        result = result + result.T
        diag_sig_sqrt = np.diag(np.reciprocal(np.sqrt(sig)))
        result = diag_sig_sqrt @ result @ diag_sig_sqrt
        result = result.T 
        return result.astype(np.float32)
    
class SPACECV():
    """
    SPACE with cross validation to select the L1 regularization parameter
    """

    def __init__(self, n_folds=4, l2_reg=0, time_series=False, alphas=None, verbose=False):
        self.n_folds = n_folds
        self.l2_reg = l2_reg
        self.time_series = time_series
        self.verbose = verbose
        self.precisions_ = {}
        if alphas is None:
            self.alphas_ = np.logspace(-3, 2)
        else:
            self.alphas_ = alphas

    def _run(self, X, l1_reg, l2_reg=0):
        s = SPACE(l1_reg, l2_reg)
        s.fit(X)
        return s.precision_, s.sig

    def likelihood_function(self, prec, cov):
        sgn, logdet = np.linalg.slogdet(prec)
        logdet = sgn * logdet
        return -logdet * np.trace(prec @ cov)

    def fit(self,X):
        """
        Runs with some cross validation
        """
        if not self.time_series:
            kf = KFold(n_splits = self.n_folds)
        else:
            kf = TimeSeriesSplit(n_splits = self.n_folds)
        l_likelihood = []
        n, p = X.shape
        # Calculate the lambdas to check
        i = 0
        for train, test in kf.split(X):
            test_errors = []

            X_train = X[train, :]
            X_test = X[test, :]
            S_test = np.cov(X_test, rowvar=False)
            outputs = Parallel(n_jobs=4)(delayed(self._run)(X_train, l, self.l2_reg) for l in self.alphas_)
            #self.precisions_[i] = precs
            self.alphas = np.logspace(-3, 2)
            for prec, sig in outputs:
                error = self.rss_function(X_test, prec, sig)
                test_errors.append(error)

            if self.verbose:
                print("Kfold %s done" % (i))

            i += 1

        min_err_i = np.argmin(test_errors)
        best_l = self.alphas_[min_err_i]

        if self.verbose:
            print("Best lambda is at %s" % best_l)

        prec = self._run(X, best_l, self.l2_reg)

        self.precision_ = prec
        self.alpha_ = best_l

    def rss_function(self, X_test, prec, sig):
        """
        An alternative approach based on how well each column predicts the others
        """
        n, p = X.shape
        rss = 0
        for i in range(p):
            rss_i = 0
            predict = 0
            vec_rss_i = 0
            for j in range(p):
                if i == j:
                    continue
                predict += prec[i, j] * np.sqrt(sig[j] / sig[i]) * X[:, j]     
            residual = np.power(X[:, i] - predict, 2).sum()
            rss += residual

        return rss

class SPACE_BIC():
    """
    Fits SPACE using BIC instead
    """
    def __init__(self, verbose=False, l2_reg=0, alphas=None):
        self.verbose = verbose
        self.outputs_ = None
        self.precision_ = None
        self.l2_reg = l2_reg
        self.alpha_ = None
        
        if alphas is None:
            self.alphas_ = np.logspace(1.8, 2.5)
        else:
            self.alphas_ = alphas

    def _run(self, X, l1_reg, l2_reg=0):
        s = SPACE(l1_reg, l2_reg)
        s.fit(X)
        return s.precision_, s.sig

    def fit(self, X):
        n, p = X.shape
        rv = norm()
        #max_l = n**(3/2) * rv.cdf(1 - (0.1/(2*p**2)))  
        #min_l = 0.01 * max_l

        if self.verbose:
            print("Lambda limits are from %s to %s" % (self.alphas_[0], self.alphas_[-1]))

        outputs = Parallel(n_jobs=4)(delayed(self._run)(X, l, self.l2_reg) for l in self.alphas_)
        bics = []
        for prec, sig in outputs:
            error = self.bic(X, prec, sig)
            bics.append(error)
        bics = np.array(bics)
        min_err_i = np.argmin(bics)
        best_l = self.alphas_[min_err_i]

        if best_l == self.alphas_[0]:
            print("WARNING: lambda is at the minimum value. It might be worth rerunning with a different set of alphas")
        
        if best_l == self.alphas_[-1]:
            print("WARNING: lambda is at the maximum value. It might be worth rerunning with a different set of alphas")

        if self.verbose:
            print("Best lambda is at %s" % best_l)
            #print(bics)
        self.alpha_ = best_l
        self.precision_ = outputs[min_err_i][0] 
        self.outputs_ = outputs       

    def bic(self, X, prec, sig):
        n, p = X.shape
        total_bic = 0
        indices = np.arange(p)
        for i in range(p):
            rss_i = 0
            predict = 0
            vec_rss_i = 0
            for j in range(p):
                if i == j:
                    continue
                #vec_rss_i += np.power(X[:, i] - prec[i, j] * np.sqrt(sig[j] / sig[i]) * X[:, j], 2).sum()
                predict += prec[i, j] * np.sqrt(sig[j] / sig[i]) * X[:, j]     
            #vec_rss_i = n*sklearn.metrics.mean_squared_error(X[:, i], predict)
            #print(rss_i)
            #print(vec_rss_i)
            residual = np.power(X[:, i] - predict, 2).sum()
            #print(vec_rss_i)

            k = np.count_nonzero(prec[indices!=i, i])
            total_bic += n * np.log(residual) + np.log(n) * k

        return total_bic

if __name__=="__main__":
    n = 10 
    p = 5
    """
    X = np.random.rand(n, p)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    #space = SPACE(0.1)
    #space.fit(X)
    #print(space.precision_)
    space_cv = SPACECV()
    space_cv.fit(X)
    print(space_cv.precision_)
    print(space_cv.alpha_)
    """
    #space_bic = SPACE_BIC(verbose=True)
    #space_bic.fit(X)
    #print(space_bic.precision_)
    #print(space_bic.alpha_)

    P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.5, smallest_coef=.4, largest_coef=.7, norm_diag=True)
    C = np.linalg.inv(P)
    X = np.random.multivariate_normal(np.zeros(p), C, n)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    S = np.cov(X.T)
    off_diag_ind = ~np.eye(p, dtype=bool)
    max_l = n*np.abs(S[off_diag_ind]).max()
    space = SPACE(max_l)
    space.fit(X)
    print(space.precision_)
    #space_cv = SPACECV()
    #space_cv.fit(X)
    #print(space_cv.precision_)
    #print(space_cv.alpha_)

    #space_bic = SPACE_BIC(verbose=True)
    #space_bic.fit(X)
    #print(space_bic.precision_)
    #print(space_bic.alpha_)

    #prec = space_r.run(X, space_cv.alpha_)
    #print(prec)