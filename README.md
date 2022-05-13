# Dynamic Bayesian mutual information (dbmi) estimation for feature selection (version 0.1)

DynamicBayesianMutualInformation estimates the posterior mean
mutual information between two binary variables under a Dirichlet 
prior over discrete states. This is performed recursively, with pseudocounts diffusing
over time according to a forgetting factor.

## Usage
```python
import numpy as np
from scipy.sparse import csr_matrix
from dbmi import DynamicBayesianMutualInformation

y = np.random.choice(2, size=(200,1))
y = np.concatenate([y, 1-y], axis=1)
X = csr_matrix(np.random.choice(2, size=(200,1000)))

dbmi = DynamicBayesianMutualInformation()
fitted = dbmi.fit(y, X)
fitted.posterior_means
```

## Parameters

### forgetting_factor : float, default=0.99

Forgetting factor applied to Dirichlet pseudocounts at each iteration.

### forward_filter : Boolean, default=False

If True, set posterior mean to zero in case there is significant evidence
that it lies below a chosen threshold.

### beta_sig : float, default = 0.05

Significance level for forward_filter.

### beta_eps : float, default = 0.00001

Threshold for forward_filter.

### n_jobs : int, default=1

Number of parallel workers to use. If RAM permits, set to number of logical processors
to maximise estimation speed.
