import numpy as np
import corner
import scipy 

from typing import Tuple, Sequence, Dict, Any, Optional


class LoadPickles:
    # Helper: extract parameter order from YAML for a given model_type
    def get_param_order_from_yaml(self, model_type: str) -> Sequence[str]:
        import yaml
        from pathlib import Path
        yml_path = Path(__file__).parent.parent / 'model' / 'brightness_models.yml'
        with open(yml_path, 'r') as f:
            config = yaml.safe_load(f)
        dists = config.get('distributions', {})
        for dist in dists.values():
            if dist.get('model_type') == model_type:
                # Return parameter order as listed in YAML
                return [k for k in dist.get('parameters', {}).keys() if k != 'type']
        return []

    # ------------------------ Sampling / I/O helpers ------------------------
    def read_quantiles(self, filename: str | None, quantiles: Sequence[float] = [0.16, 0.50, 0.84]) -> np.ndarray:
        """
        Load nested sampling results and extract weighted quantiles for each parameter.
        Returns array of shape (n_params, n_quantiles).
        """
        results = self.results['samples']
        samples = np.copy(results['samples'])

        weights = results['logwt'] - scipy.special.logsumexp(results['logwt'] - results['logz'][-1])
        weights = np.exp(weights - results['logz'][-1])
        edges = np.array([
            corner.quantile(samples[:, r], quantiles, weights=weights)
            for r in range(samples.shape[1])
        ])

        return edges
    
    # ------------------------ Main method ------------------------
    def get_parameters_from_quantiles(
        self,
        filename: str | None,
        quantiles: Sequence[float] = [0.16, 0.50, 0.84],
    ) -> Sequence[Dict[str, Any]]:
        """
        Build parameter dictionaries for each quantile, combining quantile values and fixed values.
        param_keys: list of parameter names matching quantile order (if known)
        fixed_keys: list of fixed parameter names (if known)
        cluster_params: optional cluster metadata to include
        model_type: optional model type string for template (not used here, but for future extension)
        """

        self.results = np.load(filename, allow_pickle=True)

        quantile_array = self.read_quantiles(filename, quantiles)  # shape (n_params, n_quantiles)
        param_list     = self._read_fixedvalues()
        params         = self._build_param_dicts(quantile_array, param_list)
        calibs         = self._read_calibrations(quantile_array)
        return params, calibs

    # ------------------------ building param dictionairy ------------------------
    def _read_calibrations(
        self,
        quantile_array: Optional[np.ndarray] = None,
    ) -> Sequence[Sequence[float]]:
        """
        Read measured calibration (scale) factors from the results structure.

        We expect:
        - `self.results['scales']` -> iterable of length N_scales (defaults = 1)
        - `self.results['vary'][-1]['values']['vary']` -> boolean array length N_scales indicating which are fitted
        - If scales are fitted, their fitted values are the last N_fitted rows of `quantile_array`.

        Returns a list of length n_quants; each entry is a list of length N_scales with calibrations.
        If no quantile_array is provided, returns a single-row list (defaults or fitted guesses if available).
        """
        scales = self.results['scales']
        n_scales = len(scales)

        # boolean array saying which scales were fitted
        flags = np.asarray(self.results['vary'][-1]['values']['vary'], dtype=bool)
        n_fitted = int(flags.sum())

        # If no quantiles passed or no fitted scales -> return ones
        if quantile_array is None or n_fitted == 0:
            n_quants = 1 if quantile_array is None else quantile_array.shape[1]
            return [[1.0] * n_scales for _ in range(n_quants)]

        n_quants = quantile_array.shape[1]
        calibs = np.ones((n_quants, n_scales), dtype=float)

        last_rows = quantile_array[-n_fitted:, :]  # assume last rows correspond to fitted scales
        true_indices = np.where(flags)[0]

        # align length if necessary
        min_len = min(len(true_indices), last_rows.shape[0])
        true_indices = true_indices[-min_len:]
        last_rows = last_rows[-min_len:, :]

        for r_idx, scale_idx in enumerate(true_indices):
            calibs[:, scale_idx] = last_rows[r_idx, :]

        return calibs.tolist()

    def _build_param_dicts(
        self,
        quantile_array: np.ndarray,
        param_list: Sequence[Dict[str, Any]],
    ) -> Sequence[Dict[str, Any]]:
        """
        For each quantile and each param_list element, build a dict matching parameter names to quantile values using model type and canonical order. Use fixed values for non-quantile parameters.
        Returns a list of dicts: one for each (quantile, param_list element) pair.
        """

        n_quants = quantile_array.shape[1]
        n_compts = len(param_list)
        param_dicts = [[None for _ in range(n_compts)] for _ in range(n_quants)]

        for i_quant in range(n_quants):
            for j_compt, params in enumerate(param_list):
                model_type = params['model'].get('type', None)
                model_keys = self.get_param_order_from_yaml(model_type)
                model_dict = {'type': model_type}
                idx_quant = 0
                for idx, key_param in enumerate(params['model'].keys()):
                    if key_param == 'type':
                        continue
                    key = model_keys[idx-1]
                    if params['model'][key_param]:
                        model_dict[key] = float(quantile_array[idx_quant, i_quant])
                        idx_quant += 1
                        if key == 'mass':
                            model_dict[key] = 10**model_dict[key]
                    else:
                        model_dict[key] = float(self.results['pars'][0]['model']['guess'][idx-1])
                spectrum_type = params['spectrum'].get('type', None)
                spectrum_dict = {'type': spectrum_type}
                param_dicts[i_quant][j_compt] = {'model': model_dict, 'spectrum': spectrum_dict}
        return param_dicts

    
    def _read_fixedvalues(self, ) -> Dict[str, Any]:
        """
        Reads fixed parameter values from the results dict.
        Returns a dictionary mapping parameter names to their fixed values.
        """
        fixed = []
        # Try to extract fixed values from results['vary'] if present
        for vary in self.results['vary'][:-1]:

            params = {
                'model':    {'type': vary['values']['model']['type']},
                'spectrum': {'type': vary['values']['spectrum']['type']}
            }

            for v in vary['values']:

                val = vary['values'][v].get('value', None)

                vary_list = vary['values'][v].get('vary', None)    
                for idx, p in enumerate(vary['values'][v]['vary']):
                    params[v][str(idx)] = p 

            fixed.append(params)

        return fixed

