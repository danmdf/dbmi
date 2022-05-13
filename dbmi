# dbmi (version 0.1, 13/05/2022)

import numpy as np
from scipy.special import digamma
from joblib import Parallel, delayed
from scipy.stats import beta

class DynamicBayesianMutualInformation:
    """
    Dynamic Bayesian Mutual Information estimation.
    DynamicBayesianMutualInformation estimates the posterior mean
    mutual information between two binary variables under a Dirichlet 
    prior over discrete states. This is performed recursively, with pseudocounts diffusing
    over time according to a forgetting factor.

    Parameters
    ----------
   
    forgetting_factor : float, default=0.99
        Forgetting factor applied to Dirichlet pseudocounts at each iteration.
    forward_filter : Boolean, default=False
        If True, set posterior mean to zero in case there is significant evidence
        that it lies below a chosen threshold.
    beta_sig : float, default = 0.05
        Significance level for forward_filter.
    beta_eps : float, default = 0.00001
        Threshold for forward_filter.
    n_jobs : int, default=1
        Number of parallel workers to use. If RAM permits, set to number of logical processors
        to maximise estimation speed.

    Example
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> from dbmi import DynamicBayesianMutualInformation

    >>> y = np.random.choice(2, size=(200,1))
    >>> y = np.concatenate([y, 1-y], axis=1)
    >>> X = csr_matrix(np.random.choice(2, size=(200,1000)))

    >>> dbmi = DynamicBayesianMutualInformation()
    >>> fitted = dbmi.fit(y, X)
    >>> fitted.posterior_means
    """

    def __init__(
        self,
        *,
        forgetting_factor=0.99,
        forward_filter=False,
        beta_sig=0.05,
        beta_eps=0.00001,
        n_jobs=1

    ):
        self.forgetting_factor = forgetting_factor
        self.forward_filter = forward_filter
        self.beta_sig=beta_sig
        self.beta_eps=beta_eps
        self.n_jobs = n_jobs


    def fit(self, y, X):
        """
        Estimate recursively mutual information between two-column numpy array y (representing Boolean feature)
        and each feature in scipy.sparse csr_matrix X.
        
        Parameters
        ----------
        y : numpy array of shape (T, 2)
        X : scipy.sparse matrix of shape (T, D)

        Returns
        -------
        self : object
            Fitted Estimator
        """
        # make sure it's binary, not counts
        assert X.max() == 1 
        
        # encode binary feature as two-column one-hot (slow, needs improvement)
        all_signals = []
        for i in range(X.shape[1]):
            all_signals.append(np.concatenate([X[:,i].toarray(),1-X[:,i].toarray()],axis=1))

        # now loop over all features and get posterior mean mutual information
        self.posterior_means = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._posterior_mean)(y, all_signals[j]) for j in range(len(all_signals))))

        return self


    def _posterior_mean(self, y, binary_signal_array):
        # set uniform prior        
        counts = np.ones((y.shape[1], binary_signal_array.shape[1]))

        # recursive updating of pseudocounts
        for i in range(y.shape[0]):

            # forgetting step
            counts = np.power(counts, self.forgetting_factor)

            # update pseudocounts
            counts = self._update_counts(counts, y[i], binary_signal_array[i])

        return self._eval(counts)


    def _update_counts(self, pseudocounts, new_state, new_signal):
        newcounts = pseudocounts.copy()
        newcounts[new_state == 1] += new_signal # add the actual signal obs to the pseudo count
        return newcounts


    def _eval(self, newcounts):
        """Obtain posterior mean of mutual information under Dirichlet prior over discrete states given pseudocounts"""     
        n = newcounts.sum()
        n_state = newcounts.sum(axis=1)
        n_signal = newcounts.sum(axis=0)

        post_mean = 0

        K = 0
        J = 0
        M = 0
        Q = 1

        for state in range(newcounts.shape[0]):
            n_i = n_state[state] # get state count (n_i)

            for signal in range(newcounts.shape[1]):
                n_j = n_signal[signal] # get signal count (n_j)

                n_ij = newcounts[state,signal] # get state-signal count (n_ij)

                post_mean += self._eval_ij(n_ij, n_i, n_j, n)

                K += n_ij/n * (np.log(n_ij*n/(n_i * n_j)))**2
                J += n_ij/n * np.log(n_ij*n/(n_i * n_j))
                M += (1/n_ij - 1/n_i - 1/n_j + 1/n)*n_ij*np.log(n_ij*n/(n_i*n_j))
                Q -= n_ij**2/(n_i * n_j) 

        post_mean = post_mean/n
        
        vari = (K - J**2)/(n+1) + (M + (n_state.shape[0]-1)*(n_signal.shape[0]-1)*(1/2 - J) - Q)/((n+1) * (n+2))

        # forward filter: set to zero any post_mean that is not significant at 95% level
        if self.forward_filter:
            ## first do beta approximation
            a_value = ((1-post_mean)/vari - 1/post_mean)*post_mean**2 # see 3.1 in Hutter 2002
            b_value = a_value*(1/post_mean - 1)

            # now set to zero if not significant
            if beta(a=a_value, b=b_value).cdf(self.beta_eps) > self.beta_sig:
                post_mean = 0

        return post_mean

    def _eval_ij(self, n_ij, n_i, n_j, n):
        return n_ij*(digamma(n_ij + 1) - digamma(n_i + 1) - digamma(n_j + 1) + digamma(n + 1))

