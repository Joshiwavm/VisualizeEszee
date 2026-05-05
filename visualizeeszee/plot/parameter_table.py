from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Sequence


class PlotParameterTable:
    """Mixin providing get_parameter_table() and make_parameter_table() for Manager."""

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _load_param_rows(
        self,
        fname: str,
        unit_scale: Dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Load a pickle and return a DataFrame with columns
        [median, err_lo, err_hi, frozen] indexed by '{ModelType}.{param}' strings.
        """
        params, calibs = self.get_parameters_from_quantiles(fname)
        # params[q_idx][comp_idx] = {'model': {...}, 'spectrum': {...}}
        # self.results is set as a side-effect of get_parameters_from_quantiles

        n_compts_total = len(self.results['vary']) - 1  # exclude trailing calib entry
        # Count image-domain (YAML-known) components for multi-component labelling
        n_image = sum(
            1 for j in range(n_compts_total)
            if self.get_param_order_from_yaml(
                self.results['vary'][j]['values']['model'].get('type', '')
            )
        )

        rows = []
        out_idx = 0  # index into params[q], counts only image-domain components

        for j_compt in range(n_compts_total):
            vary_entry = self.results['vary'][j_compt]
            model_type = vary_entry['values']['model'].get('type', '')
            model_keys = self.get_param_order_from_yaml(model_type)

            if not model_keys:
                # UV-plane component (pointSource etc.) — skip
                continue

            vary_flags = list(vary_entry['values']['model'].get('vary', []))
            prefix = f'{model_type}_c{j_compt}.' if n_image > 1 else f'{model_type}.'

            comp_dict = params[1][out_idx]['model'] if params[1] else {}

            for param_idx, key in enumerate(model_keys):
                is_free = bool(vary_flags[param_idx]) if param_idx < len(vary_flags) else False
                row_label = prefix + key

                try:
                    v16 = float(params[0][out_idx]['model'].get(key, np.nan))
                    v50 = float(params[1][out_idx]['model'].get(key, np.nan))
                    v84 = float(params[2][out_idx]['model'].get(key, np.nan))
                except (IndexError, TypeError, KeyError):
                    v16 = v50 = v84 = np.nan

                if unit_scale and key in unit_scale:
                    factor = unit_scale[key]
                    v16 *= factor
                    v50 *= factor
                    v84 *= factor

                rows.append({
                    'param':  row_label,
                    'median': v50,
                    'err_lo': v50 - v16,
                    'err_hi': v84 - v50,
                    'frozen': not is_free,
                })

            out_idx += 1

        # Calibration scales
        cal_flags = np.asarray(
            self.results['vary'][-1]['values'].get('vary', []), dtype=bool
        )
        for i, flag in enumerate(cal_flags):
            try:
                v16 = float(calibs[0][i])
                v50 = float(calibs[1][i])
                v84 = float(calibs[2][i])
            except (IndexError, TypeError):
                v16 = v50 = v84 = np.nan
            rows.append({
                'param':  f'calib.scale_{i}',
                'median': v50,
                'err_lo': v50 - v16,
                'err_hi': v84 - v50,
                'frozen': not bool(flag),
            })

        return pd.DataFrame(rows).set_index('param')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_parameter_table(
        self,
        fname: str,
        model_label: str | None = None,
        unit_scale: Dict[str, float] | None = None,
        save: bool = True,
        output_dir: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of posterior parameters for a single pickle.

        Parameters
        ----------
        fname : str
            Path to the dynesty pickle.
        model_label : str, optional
            Label used in the saved filename. Defaults to the pickle basename.
        unit_scale : dict {param_name: factor}, optional
            Multiply named parameters by factor before storing.
        save : bool
            Save a CSV to output_dir.
        output_dir : str, optional
            Defaults to ``../plots/VisualizeEszee/{target}/table/``.

        Returns
        -------
        pd.DataFrame
            Columns: median, err_lo, err_hi, frozen
        """
        df = self._load_param_rows(fname, unit_scale=unit_scale)

        if save:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/table/'
            os.makedirs(output_dir, exist_ok=True)
            label = model_label or os.path.splitext(os.path.basename(fname))[0]
            csv_path = os.path.join(output_dir, f'{_safe_target}_{label}.csv')
            df.to_csv(csv_path)
            print(f'Saved: {csv_path}')

        return df

    def make_parameter_table(
        self,
        fnames: Dict[str, str],
        unit_scale: Dict[str, float] | None = None,
        save: bool = True,
        output_dir: str | None = None,
        sig_figs: int = 3,
        caption: str | None = None,
    ) -> pd.DataFrame:
        """Build a multi-model parameter table suitable for LaTeX export.

        Parameters
        ----------
        fnames : dict {label: fname}
            Column header → pickle path.
        unit_scale : dict {param_name: factor}, optional
        save : bool
            Save .csv and .tex files.
        output_dir : str, optional
            Defaults to ``../plots/VisualizeEszee/{target}/table/``.
        sig_figs : int
            Significant figures used in cell strings.

        Returns
        -------
        pd.DataFrame
            Rows = parameters, columns = model labels.
            Cells: ``$x^{+hi}_{-lo}$`` (free) or ``$x$ (frozen)``.
        """
        # Default unit scaling applied before formatting (user unit_scale takes precedence)
        _DEFAULT_SCALE = {'mass': 1e-14}
        effective_scale = {**_DEFAULT_SCALE, **(unit_scale or {})}

        # Parameters to exclude from the table
        _SKIP_PARAMS = {'depth', 'log10', 'bias', 'redshift', 'alpha_p', 'temperature', 'offset'}

        # Per-parameter format: decimal places (fixed notation)
        _PARAM_FMT: Dict[str, int] = {
            'ra':    5,
            'dec':   5,
            'mass':  2,
            'e':     2,
            'angle': 0,
        }

        # Display names for the row index
        _DISPLAY_NAMES: Dict[str, str] = {
            'ra':     'RA',
            'dec':    'Dec',
            'mass':   r'M$_{500,c}$',
            'c500':   r'c$_{500}$',
            'e':      'e',
            'angle':  'PA',
            'alpha':  r'$\alpha$',
            'beta':   r'$\beta$',
            'gamma':  r'$\gamma$',
            'p_norm': r'p$_\text{norm}$',
        }

        # Parameters where a positive median gets a ~~ prefix for column alignment
        _PAD_POSITIVE = {'ra'}

        def _fmt(x: float, key: str) -> str:
            if key in _PARAM_FMT:
                return f'{x:.{_PARAM_FMT[key]}f}'
            return f'{x:.{sig_figs}g}'

        def _cell(row: pd.Series, key: str) -> str:
            pad = r'~~' if key in _PAD_POSITIVE and row['median'] >= 0 else ''
            if row['frozen']:
                return f'${pad}{_fmt(row["median"], key)}$'
            return (
                f'${pad}{_fmt(row["median"], key)}'
                f'^{{+{_fmt(row["err_hi"], key)}}}'
                f'_{{-{_fmt(row["err_lo"], key)}}}$'
            )

        raw: Dict[str, pd.DataFrame] = {}
        for label, fname in fnames.items():
            raw[label] = self._load_param_rows(fname, unit_scale=effective_scale)

        # Union of all parameter rows, preserving insertion order;
        # exclude calibration scales and unwanted A10 parameters
        all_params = list(dict.fromkeys(
            p for df in raw.values() for p in df.index
            if not p.startswith('calib.')
            and p.split('.')[-1] not in _SKIP_PARAMS
        ))

        table: Dict[str, list] = {label: [] for label in fnames}
        for param in all_params:
            key = param.split('.')[-1]
            for label, df in raw.items():
                if param in df.index:
                    table[label].append(_cell(df.loc[param], key))
                else:
                    table[label].append('—')

        display_index = [_DISPLAY_NAMES.get(p.split('.')[-1], p.split('.')[-1]) for p in all_params]
        result = pd.DataFrame(table, index=display_index)
        result.index.name = 'Parameter'

        if save:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/table/'
            os.makedirs(output_dir, exist_ok=True)

            csv_path = os.path.join(output_dir, f'{_safe_target}_parameter_table.csv')
            tex_path = os.path.join(output_dir, f'{_safe_target}_parameter_table.tex')

            result.to_csv(csv_path)
            _caption = caption if caption is not None else f'{_safe_target} model parameters'
            result.to_latex(
                tex_path,
                escape=False,
                caption=_caption,
                label=f'tab:{_safe_target}_params',
            )
            print(f'Saved: {csv_path}')
            print(f'Saved: {tex_path}')

        return result
