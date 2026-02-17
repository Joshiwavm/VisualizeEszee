import numpy as np
import corner
import scipy

from typing import Tuple, Sequence, Dict, Any, Optional

# Default reference frequency used by the eszee forward model (getinfo(reffreq=1e11)).
# Spectral model: S(nu) = amplitude * (nu / ESZEE_REF_FREQ) ** spec_index
ESZEE_REF_FREQ: float = 1e11  # 100 GHz in Hz


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
        """Build parameter dicts for image-domain (YAML-known) components only.

        UV-plane components (e.g. pointSource, gaussSource) are skipped because
        they have no image-domain representation; retrieve them via
        get_point_sources_from_pickle() instead.

        Fixes vs. original:
          * idx_quant accumulates across ALL components so free-param ordering
            in quantile_array is respected regardless of component position.
          * pars[j_compt] used (not always pars[0]) for fixed-value look-up.
          * log10-stored parameters converted via pars[j]['model']['islog'].
          * Unknown model types (not in brightness_models.yml) are skipped
            but their free-param count still advances idx_quant correctly.
        """
        n_quants = quantile_array.shape[1]

        # Pre-compute per-component info: YAML keys and number of free model params
        compt_info = []
        for j_compt, params in enumerate(param_list):
            model_type = params['model'].get('type')
            model_keys = self.get_param_order_from_yaml(model_type)
            # Count free model params (vary=True entries, excluding 'type' key)
            n_free_model = sum(
                1 for k, v in params['model'].items()
                if k != 'type' and bool(v)
            )
            # Count free spectrum params
            n_free_spec = sum(
                1 for k, v in params['spectrum'].items()
                if k != 'type' and bool(v)
            )
            compt_info.append({
                'j': j_compt,
                'params': params,
                'model_type': model_type,
                'model_keys': model_keys,
                'n_free_model': n_free_model,
                'n_free_spec': n_free_spec,
                'known': bool(model_keys),  # False for pointSource etc.
            })

        # Only include image-domain (YAML-known) components in output
        image_compts = [c for c in compt_info if c['known']]
        n_image_compts = len(image_compts)
        param_dicts = [[None for _ in range(n_image_compts)] for _ in range(n_quants)]

        for i_quant in range(n_quants):
            idx_quant = 0  # accumulates across ALL components in order
            out_idx = 0
            for ci in compt_info:
                j_compt = ci['j']
                params = ci['params']
                model_keys = ci['model_keys']
                model_type = ci['model_type']

                # islog flags from pars (tells us which params are stored as log10)
                try:
                    islog = list(self.results['pars'][j_compt]['model'].get('islog', []))
                except (IndexError, KeyError, TypeError):
                    islog = []

                if ci['known']:
                    model_dict = {'type': model_type}
                    for idx, key_param in enumerate(params['model'].keys()):
                        if key_param == 'type':
                            continue
                        param_idx = idx - 1  # 0-based after skipping 'type'
                        key = model_keys[param_idx] if param_idx < len(model_keys) else f'param_{param_idx}'
                        is_varied = bool(params['model'][key_param])
                        if is_varied:
                            val = float(quantile_array[idx_quant, i_quant])
                            # Convert log10-stored parameters back to linear
                            if param_idx < len(islog) and islog[param_idx]:
                                val = 10.0 ** val
                            model_dict[key] = val
                            idx_quant += 1
                        else:
                            try:
                                val = float(self.results['pars'][j_compt]['model']['guess'][param_idx])
                                if param_idx < len(islog) and islog[param_idx]:
                                    val = 10.0 ** val
                                model_dict[key] = val
                            except (IndexError, KeyError, TypeError):
                                model_dict[key] = 0.0

                    spectrum_type = params['spectrum'].get('type')
                    spectrum_dict = {'type': spectrum_type}
                    # Advance idx_quant for any free spectrum params (e.g. free SpecIndex)
                    for key_param, val in params['spectrum'].items():
                        if key_param == 'type':
                            continue
                        if bool(val):
                            idx_quant += 1

                    param_dicts[i_quant][out_idx] = {'model': model_dict, 'spectrum': spectrum_dict}
                    out_idx += 1
                else:
                    # Unknown type: advance idx_quant for all free params so ordering stays correct
                    idx_quant += ci['n_free_model'] + ci['n_free_spec']

        return param_dicts

    
    def get_point_sources_from_pickle(self, filename: Optional[str] = None) -> list:
        """Extract frozen point-source component parameters from the pickle.

        Point sources are components whose model type is 'pointSource' in the
        vary structure.  They are stored analytically (UV-plane), not as sky
        maps, so they are not registered via add_model.  Use the returned list
        with apply_point_source_correction() to subtract their contribution
        from the visibility residuals before deconvolution.

        Parameters
        ----------
        filename : str or None
            Path to the pickle file.  If None, re-uses self.results from the
            most recent load.

        Returns
        -------
        list of dict
            One dict per pointSource component::

                {'ra': float (deg),
                 'dec': float (deg),
                 'amplitude': float (Jy),
                 'offset': float (Jy),
                 'spec_type': str,
                 'spec_index': float,
                 'ref_freq': float (Hz)}
        """
        if filename is not None:
            self.results = np.load(filename, allow_pickle=True)

        ps_list = []
        for j_compt, vary in enumerate(self.results['vary'][:-1]):
            model_type = vary['values']['model'].get('type', '')
            if model_type != 'pointSource':
                continue
            try:
                pars_m = self.results['pars'][j_compt]['model']
                pars_s = self.results['pars'][j_compt]['spectrum']
                guess_m = pars_m['guess']
                guess_s = pars_s['guess']
            except (IndexError, KeyError):
                continue

            spec_type = vary['values']['spectrum'].get('type', 'powerLaw')
            ps_list.append({
                'ra':        float(guess_m[0]),
                'dec':       float(guess_m[1]),
                'amplitude': float(guess_m[2]),
                'offset':    float(guess_m[3]) if len(guess_m) > 3 else 0.0,
                'spec_type': spec_type,
                'spec_index': float(guess_s[0]) if len(guess_s) > 0 else 0.0,
                'ref_freq':  ESZEE_REF_FREQ,
            })

        # Cache on self so summary() can display them
        self.point_sources = ps_list
        return ps_list

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

