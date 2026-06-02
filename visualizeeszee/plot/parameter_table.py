from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Sequence


class ParameterTableResult:
    """Container for make_parameter_table output.

    Displays the nicely-formatted table in Jupyter notebooks.
    Access ``.latex`` for the LaTeX-formatted DataFrame.

    Attributes
    ----------
    display : pd.DataFrame
        Human-readable table (unicode ±, no LaTeX markup).
    latex : pd.DataFrame
        LaTeX-formatted table (``$x^{+hi}_{-lo}$`` cells).
    """

    def __init__(self, display: pd.DataFrame, latex: pd.DataFrame):
        self.display = display
        self.latex = latex

    def _repr_html_(self):
        return self.display.style.set_properties(**{
            'text-align': 'center',
            'white-space': 'nowrap',
        }).set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'th.row_heading', 'props': [('text-align', 'left')]},
        ])._repr_html_()

    def __repr__(self):
        return self.display.__repr__()

    def __str__(self):
        return self.display.__str__()


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
        name_map: Dict[str, str] | None = None,
        unit_scale: Dict[str, float] | None = None,
        center: tuple[float, float] | None = None,
        save: bool = True,
        output_dir: str | None = None,
        sig_figs: int = 3,
        caption: str | None = None,
    ) -> ParameterTableResult:
        """Build a multi-model parameter table suitable for LaTeX export.

        Rows = model runs, columns = parameters.
        Parameters from different model types sharing the same name (e.g. 'ra',
        'alpha') are merged into a single column.  Parameters unique to one
        model type (e.g. 'mass'/'c500' for A10, 'r_s' for gNFW) show '—' for
        models that do not have them.

        Also includes a Bayesian evidence column: Δln Z relative to the null
        model, plus the effective significance σ = sgn(Δln Z) √(2|Δln Z|).

        Parameters
        ----------
        fnames : dict {label: fname}
            Row label → pickle path.
        name_map : dict {label: display_name}, optional
            Rename row labels before building the table.  Keys not present in
            fnames are silently ignored.  Example::

                name_map={
                    'a10_sph':    'A10 (sph)',
                    'gnfw_abg_ell': r'gNFW $\alpha\beta\gamma$ free (ell)',
                }
        unit_scale : dict {param_name: factor}, optional
            Scaling applied before formatting (merged with defaults).
        save : bool
            Save .csv and .tex files.
        output_dir : str, optional
            Defaults to ``../plots/VisualizeEszee/{target}/table/``.
        sig_figs : int
            Significant figures used in cell strings.

        Returns
        -------
        ParameterTableResult
            ``.display``: human-readable DataFrame (renders nicely in notebooks).
            ``.latex``:   LaTeX-formatted DataFrame.
        """
        if name_map:
            fnames = {name_map.get(k, k): v for k, v in fnames.items()}

        # Default unit scaling (user unit_scale takes precedence)
        _DEFAULT_SCALE = {'mass': 1e-14, 'r_s': 3600}
        effective_scale = {**_DEFAULT_SCALE, **(unit_scale or {})}

        # Parameters to skip entirely
        _SKIP_PARAMS = {'depth', 'log10', 'bias', 'redshift', 'alpha_p',
                        'temperature', 'offset'}

        # Canonical column order
        _PARAM_ORDER = ['ra', 'dec', 'mass', 'p_norm', 'c500', 'r_s',
                        'e', 'angle', 'alpha', 'beta', 'gamma']

        # Per-parameter format: decimal places (fixed notation)
        _PARAM_FMT: Dict[str, int] = {
            'ra':    2,
            'dec':   2,
            'mass':  2,
            'e':     2,
            'angle': 0,
        }

        # LaTeX column headers
        _LATEX_NAMES: Dict[str, str] = {
            'ra':     r'$\Delta$RA ["$]$',
            'dec':    r'$\Delta$Dec ["]',
            'mass':   r'$M_{500,c}$',
            'c500':   r'$c_{500}$',
            'r_s':    r'$r_s$ ["]',
            'e':      r'$e$',
            'angle':  'PA',
            'alpha':  r'$\alpha$',
            'beta':   r'$\beta$',
            'gamma':  r'$\gamma$',
            'p_norm': r'$p_\mathrm{norm}$',
        }

        # Nice (non-LaTeX) column headers for notebook display
        _NICE_NAMES: Dict[str, str] = {
            'ra':     'ΔRA ["]',
            'dec':    'ΔDec ["]',
            'mass':   'M₅₀₀ [10¹⁴M☉]',
            'c500':   'c₅₀₀',
            'r_s':    'rₛ ["]',
            'e':      'e',
            'angle':  'PA',
            'alpha':  'α',
            'beta':   'β',
            'gamma':  'γ',
            'p_norm': 'p_norm',
        }

        _PAD_POSITIVE = set()

        def _fmt(x: float, key: str) -> str:
            if key in _PARAM_FMT:
                return f'{x:.{_PARAM_FMT[key]}f}'
            return f'{x:.{sig_figs}g}'

        def _cell_latex(row: pd.Series, key: str) -> str:
            pad = r'~~' if key in _PAD_POSITIVE and row['median'] >= 0 else ''
            if row['frozen']:
                return f'${pad}{_fmt(row["median"], key)}$'
            return (
                f'${pad}{_fmt(row["median"], key)}'
                f'^{{+{_fmt(row["err_hi"], key)}}}'
                f'_{{-{_fmt(row["err_lo"], key)}}}$'
            )

        def _cell_nice(row: pd.Series, key: str) -> str:
            val = row['median']
            if np.isnan(val):
                return '—'
            fval = _fmt(val, key)
            if row['frozen']:
                return fval
            hi, lo = row['err_hi'], row['err_lo']
            # Use ± for errors that agree within 10 %
            if (hi + lo) > 0 and abs(hi - lo) / (hi + lo) < 0.10:
                avg = (hi + lo) / 2
                return f'{fval} ± {_fmt(avg, key)}'
            return f'{fval} (+{_fmt(hi, key)} / -{_fmt(lo, key)})'

        # ------------------------------------------------------------------
        # Load posteriors
        # ------------------------------------------------------------------
        raw: Dict[str, pd.DataFrame] = {}
        for label, fname in fnames.items():
            df = self._load_param_rows(fname, unit_scale=effective_scale)
            df = df[~df.index.str.startswith('calib.')]
            df.index = pd.Index([p.split('.')[-1] for p in df.index], name='param')
            df = df[~df.index.isin(_SKIP_PARAMS)]
            df = df[~df.index.duplicated(keep='first')]
            raw[label] = df

        # ------------------------------------------------------------------
        # RA/Dec offset transform (arcsec, cos-corrected for RA)
        # ------------------------------------------------------------------
        # Determine phase centre: use supplied center or first entry's median RA/Dec
        first_df = next(iter(raw.values()))
        ra_center  = float(center[0]) if center is not None else (
            float(first_df.loc['ra', 'median']) if 'ra' in first_df.index else 0.0
        )
        dec_center = float(center[1]) if center is not None else (
            float(first_df.loc['dec', 'median']) if 'dec' in first_df.index else 0.0
        )
        cos_dec = float(np.cos(np.deg2rad(dec_center)))
        print(f'RA/Dec offsets relative to center: RA={ra_center:.5f} deg, Dec={dec_center:.5f} deg')

        for df in raw.values():
            if 'ra' in df.index:
                df.loc['ra', 'median'] = (df.loc['ra', 'median'] - ra_center) * cos_dec * 3600
                df.loc['ra', 'err_lo'] *= cos_dec * 3600
                df.loc['ra', 'err_hi'] *= cos_dec * 3600
            if 'dec' in df.index:
                df.loc['dec', 'median'] = (df.loc['dec', 'median'] - dec_center) * 3600
                df.loc['dec', 'err_lo'] *= 3600
                df.loc['dec', 'err_hi'] *= 3600

        # ------------------------------------------------------------------
        # Bayesian evidence per model
        # ------------------------------------------------------------------
        evidence: Dict[str, tuple[float, float]] = {}
        for label, fname in fnames.items():
            try:
                r = np.load(fname, allow_pickle=True)
                logz = float(np.asarray(r['samples']['logz'])[-1])
                lognull = float(r['loglnull']) if 'loglnull' in r else 0.0
                delta = logz - lognull
                sigma = float(np.sign(delta) * np.sqrt(2.0 * abs(delta)))
                evidence[label] = (delta, sigma)
            except Exception:
                evidence[label] = (np.nan, np.nan)

        # ------------------------------------------------------------------
        # Build parameter column order
        # ------------------------------------------------------------------
        all_seen = list(dict.fromkeys(
            p for df in raw.values() for p in df.index
        ))
        ordered     = [p for p in _PARAM_ORDER if p in all_seen]
        extras      = [p for p in all_seen if p not in set(ordered)]
        final_params = ordered + extras

        # ------------------------------------------------------------------
        # Build both cell grids
        # ------------------------------------------------------------------
        latex_rows: Dict[str, list] = {lbl: [] for lbl in fnames}
        nice_rows:  Dict[str, list] = {lbl: [] for lbl in fnames}

        for param in final_params:
            for lbl, df in raw.items():
                if param in df.index:
                    latex_rows[lbl].append(_cell_latex(df.loc[param], param))
                    nice_rows[lbl].append(_cell_nice(df.loc[param], param))
                else:
                    latex_rows[lbl].append('—')
                    nice_rows[lbl].append('—')

        # Add evidence column
        # Row 1: raw Δln Z vs null model.  Rows 2+: difference relative to row 1.
        ev_latex_col_hdr = r'$\Delta\ln\mathcal{Z}\ (\sigma)$'
        ev_nice_col_hdr  = 'ΔlnZ (σ)'
        _first_delta = evidence[next(iter(fnames))][0]
        for i, lbl in enumerate(fnames):
            delta, sigma = evidence[lbl]
            if np.isnan(delta):
                latex_rows[lbl].append('—')
                nice_rows[lbl].append('—')
            elif i == 0:
                latex_rows[lbl].append(f'${delta:.1f}\\ ({sigma:.1f}\\sigma)$')
                nice_rows[lbl].append(f'{delta:.1f} ({sigma:.1f}σ)')
            else:
                rel = delta - _first_delta
                rel_sigma = float(np.sign(rel) * np.sqrt(2.0 * abs(rel)))
                latex_rows[lbl].append(f'${rel:+.1f}\\ ({rel_sigma:+.1f}\\sigma)$')
                nice_rows[lbl].append(f'{rel:+.1f} ({rel_sigma:+.1f}σ)')

        latex_cols = [_LATEX_NAMES.get(p, p) for p in final_params] + [ev_latex_col_hdr]
        nice_cols  = [_NICE_NAMES.get(p, p)  for p in final_params] + [ev_nice_col_hdr]

        latex_df = pd.DataFrame(latex_rows, index=latex_cols).T
        latex_df.index.name = 'Model'
        nice_df  = pd.DataFrame(nice_rows,  index=nice_cols).T
        nice_df.index.name  = 'Model'

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        if save:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/table/'
            os.makedirs(output_dir, exist_ok=True)

            csv_path = os.path.join(output_dir, f'{_safe_target}_parameter_table.csv')
            tex_path = os.path.join(output_dir, f'{_safe_target}_parameter_table.tex')

            nice_df.to_csv(csv_path)
            _caption = caption if caption is not None else f'{_safe_target} model parameters'
            latex_df.to_latex(
                tex_path,
                escape=False,
                caption=_caption,
                label=f'tab:{_safe_target}_params',
            )
            print(f'Saved: {csv_path}')
            print(f'Saved: {tex_path}')

        return ParameterTableResult(display=nice_df, latex=latex_df)
