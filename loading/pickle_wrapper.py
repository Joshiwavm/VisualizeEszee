import numpy as np
import corner
import scipy 

from typing import Tuple, Sequence, Dict, Any, Optional


class LoadPickles:

    # ------------------------ Sampling / I/O helpers ------------------------
    def get_parameters_from_quantiles(self, filename: str | None, quantiles: Sequence[float] = [0.16, 0.50, 0.84]):
        """Load nested sampling results and extract weighted quantiles.

        Note: expects an npz with entries similar to the original codebase.
        """

        results = np.load(filename, allow_pickle=True)
        results = results['samples']
        samples = np.copy(results['samples'])

        weights = results['logwt'] - scipy.special.logsumexp(results['logwt'] - results['logz'][-1])
        weights = np.exp(weights - results['logz'][-1])
        edges = np.array([
            corner.quantile(samples[:, r], quantiles, weights=weights)
            for r in range(samples.shape[1])
        ])
        return edges.T