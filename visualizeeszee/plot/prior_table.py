from __future__ import annotations

import io
import os
import contextlib
import numpy as np
import pandas as pd
from typing import Dict


class PriorTableResult:
    """Container for make_prior_table output.

    Displays the nicely-formatted table in Jupyter notebooks.
    Access ``.latex`` for the LaTeX-formatted DataFrame.

    Attributes
    ----------
    display : pd.DataFrame
        Human-readable table (unicode, no LaTeX markup).
    latex : pd.DataFrame
        LaTeX-formatted table.
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


class PlotPriorTable:
    """Mixin providing make_prior_table() for Manager.

    Prior ranges are not stored in the eszee result pickles (they are popped
    from the parameter list before dumping), so priors are read from the eszee
    init script that configured each run.  The script is executed up to (but
    excluding) the ``eszee.build`` call, which is safe: everything before it is
    plain numpy/astropy parameter bookkeeping.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_init_namespace(init_path: str) -> dict:
        """Execute an eszee init script up to the sampler construction and
        return its namespace (contains ``parlist`` and ``vislist``)."""
        with open(init_path) as f:
            lines = f.read().split('\n')

        cut = len(lines)
        for i, line in enumerate(lines):
            if 'eszee.build' in line:
                cut = i
                break
        kept = []
        for line in lines[:cut]:
            # eszee itself is not needed to build the parameter list
            if line.strip() == 'import eszee' or line.strip().startswith('import eszee '):
                continue
            kept.append(line)

        ns: dict = {'__name__': f'_prior_parse_{os.path.basename(init_path)}'}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile('\n'.join(kept), init_path, 'exec'), ns)

        if 'parlist' not in ns:
            raise ValueError(f'No parlist found in init script: {init_path}')
        return ns

    def _load_prior_rows(self, init_path: str) -> pd.DataFrame:
        """Parse one init script and return a DataFrame with columns
        [nice, latex] indexed by parameter name."""

        _SKIP_PARAMS = {'offset', 'temperature', 'bias', 'log10', 'depth'}
        # Parameters whose uniform bounds are reported as offsets from the
        # guess, in arcsec (RA additionally cos(Dec)-corrected)
        _AS_OFFSET = {'ra', 'dec'}
        # Linear unit scaling applied after any 10** transform
        _UNIT_SCALE = {'mass': 1e-14, 'r_s': 3600.0}

        ns = self._load_init_namespace(init_path)

        rows = []

        def _add(param, nice, latex):
            rows.append({'param': param, 'nice': nice, 'latex': latex})

        def _fmt(x):
            if not np.isfinite(x):
                return 'inf' if x > 0 else '-inf'
            return f'{x:.3g}'

        for compt in ns['parlist']:
            model = compt['model']
            model_type = model['type']
            keys = self.get_param_order_from_yaml(model_type)
            if not keys:
                # UV-plane component (pointSource etc.) — skip
                continue

            guess = np.asarray(model['guess'], dtype=float)
            prange = model['range']
            prior = np.asarray(model['prior'])
            islog = np.asarray(model.get('islog', np.zeros(guess.shape[0], dtype=bool)))

            dec_ctr = None
            if 'dec' in keys[:2]:
                dec_ctr = guess[keys.index('dec')]
            cos_dec = float(np.cos(np.deg2rad(dec_ctr))) if dec_ctr is not None else 1.0

            for i, key in enumerate(keys[:guess.shape[0]]):
                if key in _SKIP_PARAMS:
                    continue

                scale = _UNIT_SCALE.get(key, 1.0)
                lo, hi = float(prange[i][0]), float(prange[i][1])
                ptype = str(prior[i])

                if ptype == 'd' or lo == hi:
                    # eszee freezes any parameter whose range collapses
                    val = 10**guess[i] if islog[i] else guess[i]
                    _add(key, _fmt(val * scale), f'${_fmt(val * scale)}$')
                elif ptype in ('u', 'ud', 'ur'):
                    if islog[i]:
                        a, b = 10**lo * scale, 10**hi * scale
                        _add(key, f'log-U({_fmt(a)}, {_fmt(b)})',
                             rf'log-$\mathcal{{U}}({_fmt(a)}, {_fmt(b)})$')
                    else:
                        a, b = lo * scale, hi * scale
                        if key in _AS_OFFSET:
                            factor = 3600.0 * (cos_dec if key == 'ra' else 1.0)
                            a = (lo - guess[i]) * factor
                            b = (hi - guess[i]) * factor
                        _add(key, f'U({_fmt(a)}, {_fmt(b)})',
                             rf'$\mathcal{{U}}({_fmt(a)}, {_fmt(b)})$')
                elif ptype == 'g':
                    # eszee convention: range = [mu - sigma, mu + sigma]
                    mu = 0.5 * (hi + lo) * scale
                    sig = 0.5 * (hi - lo) * scale
                    _add(key, f'N({_fmt(mu)}, {_fmt(sig)})',
                         rf'$\mathcal{{N}}({_fmt(mu)}, {_fmt(sig)})$')
                elif ptype == 'tg':
                    # eszee convention: range = [mu - sigma, mu + sigma, lo, hi]
                    mu = 0.5 * (hi + lo) * scale
                    sig = 0.5 * (hi - lo) * scale
                    a, b = float(prange[i][2]) * scale, float(prange[i][3]) * scale
                    _add(key, f'TN({_fmt(mu)}, {_fmt(sig)}; [{_fmt(a)}, {_fmt(b)}])',
                         rf'$\mathcal{{N}}_{{[{_fmt(a)},{_fmt(b)}]}}({_fmt(mu)}, {_fmt(sig)})$')
                elif ptype == 'e':
                    a, b = float(prange[i][0]) * scale, float(prange[i][1]) * scale
                    _add(key, f'Exp({_fmt(a)}, {_fmt(b)})',
                         rf'$\mathrm{{Exp}}({_fmt(a)}, {_fmt(b)})$')
                elif ptype == 'i':
                    _add(key, 'improper', 'improper')
                else:
                    _add(key, ptype, ptype)

        # Flux-calibration priors from vislist (one per interferometric data set)
        cal_cells = []
        for vis in ns.get('vislist', []):
            sc = vis.get('scale')
            if sc is None:
                continue
            sguess, srange, sptype = float(sc[0]), sc[1], str(sc[2])
            slo, shi = float(srange[0]), float(srange[1])
            if slo == shi:
                cal_cells.append((_fmt(sguess), f'${_fmt(sguess)}$'))
            elif sptype == 'g':
                # eszee convention: range = [mu - sigma, mu + sigma]
                mu, sig = 0.5 * (shi + slo), 0.5 * (shi - slo)
                cal_cells.append((f'N({_fmt(mu)}, {_fmt(sig)})',
                                  rf'$\mathcal{{N}}({_fmt(mu)}, {_fmt(sig)})$'))
            elif sptype in ('u', 'ud', 'ur'):
                cal_cells.append((f'U({_fmt(slo)}, {_fmt(shi)})',
                                  rf'$\mathcal{{U}}({_fmt(slo)}, {_fmt(shi)})$'))
            else:
                cal_cells.append((sptype, sptype))
        if cal_cells:
            if len(set(cal_cells)) == 1:
                _add('calib', *cal_cells[0])
            else:
                for i, cell in enumerate(cal_cells):
                    _add(f'calib_{i}', *cell)

        df = pd.DataFrame(rows).set_index('param')
        return df[~df.index.duplicated(keep='first')]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def make_prior_table(
        self,
        inits: Dict[str, str],
        name_map: Dict[str, str] | None = None,
        save: bool = True,
        output_dir: str | None = None,
        caption: str | None = None,
    ) -> PriorTableResult:
        """Build a prior-summary table from eszee init scripts.

        Rows = parameters, columns = model runs.  Fixed parameters show their
        value; sampled parameters show the prior distribution.  Parameters not
        present in a run show '—'.  RA/Dec uniform priors are converted to
        arcsec offsets around the prior centre; log-sampled parameters are
        reported as log-uniform over the corresponding linear range (mass in
        1e14 Msun, r_s in arcsec).

        Parameters
        ----------
        inits : dict {label: init_path}
            Column label → path to the eszee init script of that run.
        name_map : dict {label: display_name}, optional
            Rename column labels, same convention as make_parameter_table.
        save : bool
            Save .csv and .tex files.
        output_dir : str, optional
            Defaults to ``../plots/VisualizeEszee/{target}/table/``.
        caption : str, optional
            Caption for the LaTeX table.

        Returns
        -------
        PriorTableResult
            ``.display``: human-readable DataFrame.
            ``.latex``:   LaTeX-formatted DataFrame.
        """
        if name_map:
            inits = {name_map.get(k, k): v for k, v in inits.items()}

        # Canonical row order (calib rows always last)
        _PARAM_ORDER = ['ra', 'dec', 'mass', 'p_norm', 'c500', 'r_s',
                        'e', 'angle', 'alpha', 'beta', 'gamma', 'alpha_p',
                        'redshift']

        _LATEX_NAMES: Dict[str, str] = {
            'ra':       r'$\Delta$RA ["]',
            'dec':      r'$\Delta$Dec ["]',
            'mass':     r'$M_{500,c}$ [$10^{14}$M$_\odot$]',
            'c500':     r'$c_{500}$',
            'r_s':      r'$r_s$ ["]',
            'e':        r'$e$',
            'angle':    'PA [deg]',
            'alpha':    r'$\alpha$',
            'beta':     r'$\beta$',
            'gamma':    r'$\gamma$',
            'p_norm':   r'$p_\mathrm{norm}$',
            'alpha_p':  r'$\alpha_\mathrm{P}$',
            'redshift': r'$z$',
            'calib':    r'$f_\mathrm{cal}$',
        }

        _NICE_NAMES: Dict[str, str] = {
            'ra':       'ΔRA ["]',
            'dec':      'ΔDec ["]',
            'mass':     'M₅₀₀ [10¹⁴M☉]',
            'c500':     'c₅₀₀',
            'r_s':      'rₛ ["]',
            'e':        'e',
            'angle':    'PA [deg]',
            'alpha':    'α',
            'beta':     'β',
            'gamma':    'γ',
            'p_norm':   'p_norm',
            'alpha_p':  'αₚ',
            'redshift': 'z',
            'calib':    'f_cal',
        }

        raw: Dict[str, pd.DataFrame] = {
            label: self._load_prior_rows(path) for label, path in inits.items()
        }

        all_seen = list(dict.fromkeys(p for df in raw.values() for p in df.index))
        ordered = [p for p in _PARAM_ORDER if p in all_seen]
        extras = [p for p in all_seen if p not in set(ordered) and not p.startswith('calib')]
        calibs = [p for p in all_seen if p.startswith('calib')]
        final_params = ordered + extras + calibs

        nice_grid: Dict[str, list] = {}
        latex_grid: Dict[str, list] = {}
        for label, df in raw.items():
            nice_grid[label] = [
                df.loc[p, 'nice'] if p in df.index else '—' for p in final_params
            ]
            latex_grid[label] = [
                df.loc[p, 'latex'] if p in df.index else '—' for p in final_params
            ]

        nice_idx = [_NICE_NAMES.get(p, p) for p in final_params]
        latex_idx = [_LATEX_NAMES.get(p, p) for p in final_params]

        nice_df = pd.DataFrame(nice_grid, index=nice_idx)
        nice_df.index.name = 'Parameter'
        latex_df = pd.DataFrame(latex_grid, index=latex_idx)
        latex_df.index.name = 'Parameter'

        if save:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/table/'
            os.makedirs(output_dir, exist_ok=True)

            csv_path = os.path.join(output_dir, f'{_safe_target}_prior_table.csv')
            tex_path = os.path.join(output_dir, f'{_safe_target}_prior_table.tex')

            nice_df.to_csv(csv_path)
            _caption = caption if caption is not None else f'{_safe_target} model priors'
            latex_df.to_latex(
                tex_path,
                escape=False,
                caption=_caption,
                label=f'tab:{_safe_target}_priors',
            )
            print(f'Saved: {csv_path}')
            print(f'Saved: {tex_path}')

        return PriorTableResult(display=nice_df, latex=latex_df)
