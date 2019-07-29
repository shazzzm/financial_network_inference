import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from sklearn.preprocessing import StandardScaler, normalize
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.stats import norm
import sklearn.datasets
import space_r
import sklearn.linear_model as lm
import scipy

class SPACE():
    """
    Python implementation of SPACE. We use the C version of the Joint Sparse Regression model 
    """
    def __init__(self, l1_reg, l2_reg=0, sig=None, weight=None, verbose=False):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.l1_reg_ctype = ctypes.c_float(self.l1_reg)
        self.l2_reg_ctype = ctypes.c_float(self.l2_reg)

        self.sig_ = sig
        self.weight = weight

        self.lib = ctypes.CDLL("jsrm.so")   
        self.fun = self.lib.JSRM
        self.fun.restype = None
        self.fun.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), 
            ctypes.POINTER(ctypes.c_float), ndpointer(ctypes.c_float),ndpointer(ctypes.c_float), ctypes.POINTER(ctypes.c_int), 
            ctypes.POINTER(ctypes.c_int), ndpointer(ctypes.c_float)]
        self.precision_ = None
        self.verbose = verbose


    def run_jsrm(self, X):
        X = X.copy()
        n, p = X.shape
        n_in = ctypes.c_int(n)
        p_in = ctypes.c_int(p)
        #self.sig = np.array(self.sig)
        sigma_sr = np.sqrt(self.sig_).astype(np.float32)
        n_iter = ctypes.c_int(500)
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
        if self.sig_ is None:
            update_sig = True
            self.sig_ = np.ones(p, dtype=np.float32)

        for i in range(iter):
            space_prec = self.run_jsrm(X)
            np.fill_diagonal(space_prec, 1)
            ind = np.triu_indices(p)
            coef = space_prec[ind]
            beta = self.Beta_coef(coef, self.sig_)
        
            if not update_sig:
                break
            else: ## update sig or weight
                self.sig_ = self.InvSig(X, beta)   
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
        result = np.power(residue, 2).mean(axis=0)
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
        reciprocal_diag_sig_sqrt = np.diag(np.reciprocal(np.sqrt(sig)))
        diag_sig_sqrt = np.diag(np.sqrt(sig))
        result = reciprocal_diag_sig_sqrt @ result @ diag_sig_sqrt
        result = result.T 
        return result.astype(np.float32)

    def build_input(self, X):
        """
        Builds the inputs into the lasso regression problem
        """
        n, p = X.shape
        new_n = n * p
        new_p = int(p * (p - 1)/2)
        #new_X = np.zeros((new_n, new_p))
        new_X = scipy.sparse.dok_matrix((new_n, new_p))
        #new_X = sp.sparse.lil_matrix((new_n, new_p))
        indices = np.arange(p)

        if self.sig_ is None:
            self.sig_ = np.ones(p)

        x = 0
        for i in range(p):
            for j in range(i+1, p):
                #if i == j:
                #    continue
                new_col = np.zeros((n*p, 1))
                new_col[i*n:(i+1)*n] = np.sqrt(self.sig_[j]/self.sig_[i]) * X[:, j].reshape((n, 1))
                new_col[j*n:(j+1)*n] = np.sqrt(self.sig_[i]/self.sig_[j]) * X[:, i].reshape((n, 1))

                new_X[:, x] = new_col
                x += 1
        new_y = X.flatten(order='F')

        return new_X.tocoo(), new_y

    def python_solve(self, X, iter=2):
        """
        Builds and solves the model in pure python
        """
        n, p = X.shape
        space_prec = np.zeros((p, p))
        self.sig_ = np.ones(p, dtype=np.float32)
        ind = np.triu_indices(p, k=1)
        ind_2 = np.triu_indices(p)
        #X = normalize(X, axis=1)

        for i in range(iter):
            X_inp, y_inp = self.build_input(X)
            lasso = lm.Lasso(self.l1_reg/(X_inp.shape[0]))
            lasso.fit(X_inp, y_inp)
            coef = lasso.coef_
            space_prec[ind] = coef
            space_prec += space_prec.T
            np.fill_diagonal(space_prec, 1)

            coef = space_prec[ind_2]
            beta = self.Beta_coef(coef, self.sig_)
            self.sig_ = self.InvSig(X, beta)   
            print(self.sig_)
        return space_prec


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
            self.alphas_ = np.logspace(-3, 3, 50)
        else:
            self.alphas_ = alphas

    def _run(self, X, l1_reg, l2_reg=0):
        if self.verbose:
            print("Running %s" % l1_reg)
        s = SPACE(l1_reg, l2_reg)
        s.fit(X)
        return s.precision_, s.sig_

    def fit(self, X):
        n, p = X.shape
        rv = norm()
        #max_l = n**(3/2) * rv.cdf(1 - (0.1/(2*p**2)))  
        #min_l = 0.01 * max_l

        rerun = True

        min_l = 1
        max_l = 200

        while rerun:
            self.alphas_ = np.linspace(min_l, max_l)
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
            rerun = False
            if best_l == self.alphas_[0]:
                rerun = True
                print("WARNING: lambda is at the minimum value. It might be worth rerunning with a different set of alphas")
                min_l = min_l*0.1
                max_l = max_l*0.1
            
            if best_l == self.alphas_[-1]:
                rerun = True
                print("WARNING: lambda is at the maximum value. It might be worth rerunning with a different set of alphas")
                min_l = min_l*10
                max_l = max_l*10

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
                predict += prec[i, j] * np.sqrt(sig[j] / sig[i]) * X[:, j]     
            residual = np.power(X[:, i] - predict, 2).sum()

            k = np.count_nonzero(prec[indices!=i, i])
            total_bic += n * np.log(residual) + np.log(n) * k

        return total_bic

class SPACE_BIC_Python(SPACE_BIC):
    """
    BIC version that fits using Python implementation
    """
    def __init__(self, verbose=False, l2_reg=0, alphas=None):
        self.verbose = verbose
        self.outputs_ = None
        self.precision_ = None
        self.l2_reg = l2_reg
        self.alpha_ = None
        self.alphas_ = None

    def _run(self, X, l1_reg, l2_reg=0):
        s = SPACE(l1_reg)
        prec = s.python_solve(X, l1_reg)

        return prec, s.sig_

    def fit(self, X):
        n, p = X.shape
        #max_l = n**(3/2) * rv.cdf(1 - (0.1/(2*p**2)))  
        #min_l = 0.01 * max_l
        s = SPACE(1)
        new_X, new_y = s.build_input(X)

        if self.alphas_ is None:
            max_l = np.abs(new_X.T @ new_y).max()
            min_l = 0.0001 * max_l

        rerun = True

        while rerun:
            self.alphas_ = np.logspace(np.log10(min_l), np.log10(max_l))
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
            rerun = False
            if best_l == self.alphas_[0]:
                rerun = True
                print("WARNING: lambda is at the minimum value. It might be worth rerunning with a different set of alphas")
                min_l = min_l*0.1
                max_l = max_l*0.1
            
            if best_l == self.alphas_[-1]:
                rerun = True
                print("WARNING: lambda is at the maximum value. It might be worth rerunning with a different set of alphas")
                min_l = min_l*10
                max_l = max_l*10

        if self.verbose:
            print("Best lambda is at %s" % best_l)
            #print(bics)
        self.alpha_ = best_l
        self.precision_ = outputs[min_err_i][0] 
        self.outputs_ = outputs       

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

    P = sklearn.datasets.make_sparse_spd_matrix(dim=p, alpha=0.7, smallest_coef=.4, largest_coef=.7, norm_diag=True)
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
    print(space.python_solve(X))

    space_bic = SPACE_BIC_Python(verbose=True)
    space_bic.fit(X)
    print(space_bic.precision_)

    #space_cv = SPACECV()
    #space_cv.fit(X)
    #print(space_cv.precision_)
    #print(space_cv.alpha_)

    #space_bic = SPACE_BIC(verbose=True)
    #space_bic.fit(X)
    #print(space_bic.precision_)
    #print(space_bic.alpha_)

    prec = space_r.run(X, max_l)
    print(prec[0])