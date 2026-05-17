"""Corner plot mixin for dynesty nested-sampling pickle files.

Supports two backends selectable via the `backend` argument:
  - 'corner'   : uses the corner package (corner.corner)
  - 'dynesty'  : uses dynesty.plotting.cornerplot (no extra dependency)

Auto-labels are built from the pickle's vary structure + brightness_models.yml
parameter ordering, so you rarely need to pass labels manually.
"""
from __future__ import annotations

import os
import re
import warnings
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from ..utils.style import setup_plot_style


class _FigProxy:
    """Thin wrapper that exposes a subset of axes as if it were a full Figure.

    Passed as ``fig=`` to ``corner.corner`` so it draws on a K×K subgrid of
    the main figure without creating a new figure or copying artists.
    Corner only needs ``fig.axes`` (list) plus attribute delegation for the
    rest — this satisfies both requirements.
    """
    def __init__(self, real_fig, axes_subset: list):
        self._fig = real_fig
        self.axes = list(axes_subset)

    def __getattr__(self, name: str):
        return getattr(self._fig, name)


# Human-readable LaTeX labels for common YAML parameter names
_PARAM_LABELS: dict[str, str] = {
    'ra':          r'RA [deg]',
    'dec':         r'Dec [deg]',
    'p_norm':      r'$P_0$ [keV cm$^{-3}$]',
    'r_s':         r'$r_s$ [deg]',
    'e':           r'$e$',
    'angle':       r'$\theta$ [deg]',
    'offset':      r'offset',
    'temperature': r'$T_e$ [keV]',
    'alpha':       r'$\alpha$',
    'beta':        r'$\beta$',
    'gamma':       r'$\gamma$',
    'redshift':    r'$z$',
    'mass':        r'$M_{500}$ [$M_\odot$]',
    'log10m':      r'$\log_{10}M_{500}$',
    'c500':        r'$c_{500}$',
    'alpha_p':     r'$\alpha_p$',
    'bias':        r'$b$',
    # point-source params (positional order in guess array)
    'ps_ra':       r'RA$_\mathrm{PS}$ [deg]',
    'ps_dec':      r'Dec$_\mathrm{PS}$ [deg]',
    'ps_amp':      r'$S_\nu$ [Jy]',
    'ps_offset':   r'offset$_\mathrm{PS}$',
    'ps_specidx':  r'$\alpha_{sl}$',
}

_PS_PARAM_NAMES = ['ps_ra', 'ps_dec', 'ps_amp', 'ps_offset', 'ps_specidx']

# Label overrides when a param is scaled by a known factor
_UNIT_SCALE_LABELS: dict[str, dict[float, str]] = {
    'r_s':  {60.0: r'$r_s$ [arcmin]', 3600.0: r"$r_s$ ['']"},
    'ra':   {60.0: r'RA [arcmin]'},
    'dec':  {60.0: r'Dec [arcmin]'},
    'angle': {1.0: r'$\theta$ [deg]'},
}

_DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Sentinel for frozen-param columns: far below any real value and below any span range.
_FROZEN_SENTINEL = -1e30
_SENTINEL_THRESHOLD = -0.5e30   # anything below this is sentinel


class PlotCorner:
    """Mixin providing plot_corner() for Manager."""

    # ------------------------------------------------------------------
    # Param name / label helpers
    # ------------------------------------------------------------------
    def _get_corner_names_and_labels(self, results) -> tuple[list[str], list[str]]:
        """Return (raw_names, display_labels) for all FREE parameters.

        raw_names: YAML keys (e.g. 'r_s', 'p_norm') with '_c{j}' suffix for
        multi-component models.  display_labels: corresponding LaTeX strings.
        """
        raw_names: list[str] = []
        labels: list[str] = []
        vary_list = list(results['vary'][:-1])
        # Count only non-pointSource components that have ≥1 free model param
        n_model_compts = sum(
            1 for v in vary_list
            if v['values']['model'].get('type', '') != 'pointSource'
            and any(v['values']['model'].get('vary', []))
        )

        for j, vary in enumerate(vary_list):
            model_type = vary['values']['model'].get('type', '')
            suffix = f' (c{j})' if n_model_compts > 1 else ''
            raw_suffix = f'_c{j}' if n_model_compts > 1 else ''

            if model_type == 'pointSource':
                param_names = _PS_PARAM_NAMES
            else:
                param_names = list(self.get_param_order_from_yaml(model_type))

            model_vary_flags = vary['values']['model'].get('vary', [])
            for idx, is_varied in enumerate(model_vary_flags):
                if is_varied:
                    raw = param_names[idx] if idx < len(param_names) else f'param_{idx}'
                    raw_names.append(raw.lower() + raw_suffix)
                    labels.append(_PARAM_LABELS.get(raw.lower(), raw) + suffix)

            spec_vary_flags = vary['values']['spectrum'].get('vary', [])
            for idx, is_varied in enumerate(spec_vary_flags):
                if is_varied:
                    raw_names.append('ps_specidx' + raw_suffix)
                    labels.append(_PARAM_LABELS.get('ps_specidx', f'spec_{idx}') + suffix)

        cal_flags = np.asarray(results['vary'][-1]['values'].get('vary', []), dtype=bool)
        scale_names = getattr(results, 'get', lambda k, d: d)('scale_names', None)
        for i, is_varied in enumerate(cal_flags):
            if is_varied:
                raw_names.append(f'cal_{i}')
                name = (scale_names[i] if scale_names and i < len(scale_names)
                        else f'$\\alpha_{{cal,{i}}}$')
                labels.append(name)

        return raw_names, labels

    def _get_corner_islog_flags(self, results) -> list[bool]:
        """Return islog bool per FREE parameter column (same order as samples).

        Calibration params are never log-stored, so they always append False.
        """
        flags: list[bool] = []
        vary_list = list(results['vary'][:-1])

        for j, vary in enumerate(vary_list):
            model_type = vary['values']['model'].get('type', '')
            if model_type == 'pointSource':
                islog = []
            else:
                try:
                    islog = list(results['pars'][j]['model'].get('islog', []))
                except (IndexError, KeyError, TypeError):
                    islog = []

            model_vary_flags = vary['values']['model'].get('vary', [])
            for idx, is_varied in enumerate(model_vary_flags):
                if is_varied:
                    flags.append(bool(islog[idx]) if idx < len(islog) else False)

            spec_vary_flags = vary['values']['spectrum'].get('vary', [])
            for _, is_varied in enumerate(spec_vary_flags):
                if is_varied:
                    flags.append(False)

        cal_flags = np.asarray(results['vary'][-1]['values'].get('vary', []), dtype=bool)
        flags.extend([False] * int(cal_flags.sum()))
        return flags

    def _get_corner_frozen_values(self, results) -> dict[str, float]:
        """Return {raw_name: value} for all FROZEN model parameters.

        Values are read from results['pars'][j]['model']['guess'] and
        log10-converted where results['pars'][j]['model']['islog'] is True.
        """
        frozen: dict[str, float] = {}
        vary_list = list(results['vary'][:-1])
        n_model_compts = sum(
            1 for v in vary_list
            if v['values']['model'].get('type', '') != 'pointSource'
            and any(v['values']['model'].get('vary', []))
        )

        for j, vary in enumerate(vary_list):
            model_type = vary['values']['model'].get('type', '')
            raw_suffix = f'_c{j}' if n_model_compts > 1 else ''

            if model_type == 'pointSource':
                param_names = _PS_PARAM_NAMES
            else:
                param_names = list(self.get_param_order_from_yaml(model_type))

            model_vary_flags = vary['values']['model'].get('vary', [])
            try:
                guess = list(results['pars'][j]['model']['guess'])
                islog = list(results['pars'][j]['model'].get('islog', []))
            except (IndexError, KeyError, TypeError):
                guess, islog = [], []

            for idx, is_varied in enumerate(model_vary_flags):
                if not is_varied and idx < len(guess):
                    raw = param_names[idx] if idx < len(param_names) else f'param_{idx}'
                    val = float(guess[idx])
                    if idx < len(islog) and islog[idx]:
                        val = 10.0 ** val
                    frozen[raw.lower() + raw_suffix] = val

        return frozen

    def _get_corner_labels(self, results) -> list[str]:
        """Backwards-compatible: return only display labels."""
        _, labels = self._get_corner_names_and_labels(results)
        return labels

    @staticmethod
    def _param_matches(raw_name: str, param_filter: list[str]) -> bool:
        """True if raw_name matches any entry in param_filter.

        Handles multi-component suffix: 'r_s_c0' matches filter entry 'r_s'.
        """
        if raw_name in param_filter:
            return True
        base = re.sub(r'_c\d+$', '', raw_name)
        return base in param_filter

    def _resolve_corner_input(self, name_or_path: str) -> tuple[str, str]:
        """Resolve a model name or raw filepath to (filepath, display_label).

        Lookup order:
        1. Exact key in self.models  (e.g. 'a10_dynesty_q0.5')
        2. Prefix match '_q*'        (e.g. 'a10_dynesty' → 'a10_dynesty_q0.5')
        3. Raw filepath              (label = basename without extension)
        """
        models = getattr(self, 'models', {})

        if name_or_path in models:
            fname = models[name_or_path].get('filename')
            if fname:
                return fname, name_or_path

        prefix = name_or_path + '_q'
        for key, info in models.items():
            if key.startswith(prefix):
                fname = info.get('filename')
                if fname:
                    return fname, name_or_path  # use the short name as label

        # Fall back: treat as a filepath; shorten label to basename stem
        label = os.path.splitext(os.path.basename(name_or_path))[0]
        return name_or_path, label

    def _load_corner_pickle(self, filename: str) -> tuple:
        """Load one pickle.

        Returns
        -------
        (free_names, frozen_names_values, labels, samples, weights, sres, results)

        free_names  : list[str]  raw names of FREE params (columns of samples)
        frozen_vals : dict[str, float]  raw_name → frozen value
        labels      : list[str]  display labels matching free_names
        samples     : ndarray (n_live, n_free)
        weights     : ndarray (n_live,)
        sres        : dict  (results['samples'])
        results     : NpzFile
        """
        results = np.load(filename, allow_pickle=True)
        sres = results['samples']
        samples = np.copy(np.asarray(sres['samples']))
        logwt = np.asarray(sres['logwt'])
        logz = np.asarray(sres['logz'])
        weights = np.exp(logwt - scipy.special.logsumexp(logwt - logz[-1]) - logz[-1])
        free_names, labels = self._get_corner_names_and_labels(results)
        frozen_vals = self._get_corner_frozen_values(results)
        islog_flags = self._get_corner_islog_flags(results)
        n_free = len(labels)
        samples = samples[:, :n_free]
        for col, flag in enumerate(islog_flags[:n_free]):
            if flag:
                samples[:, col] = 10.0 ** samples[:, col]
        return free_names, frozen_vals, labels, samples, weights, sres, results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_corner(
        self,
        filename: str | list[str],
        labels: list[str] | None = None,
        params: list[str] | None = None,
        backend: str = 'corner',
        unit_scale: dict[int, float] | None = None,
        unit_scale_by_name: dict[str, float] | None = None,
        colors: list | None = None,
        save_plots: bool = False,
        output_dir: str | None = None,
        use_style: bool = True,
        n_sigma_range: float = 4.0,
        return_fig: bool = False,
        **kwargs,
    ):
        """Make a weighted corner plot from one or more dynesty pickle files.

        Parameters
        ----------
        filename : str or list of str
            Path(s) to dynesty `.npz` pickle(s).  When a list is given the
            posteriors are overplotted.  By default all parameters that are
            free in **at least one** file are shown; files that have a param
            frozen show a vertical dashed line at the frozen value on the
            diagonal panel instead of a posterior.
        labels : list of str, optional
            Override axis labels for the displayed parameters.
        params : list of str, optional
            Restrict to a subset of YAML parameter names
            (e.g. ``['r_s', 'p_norm']``).  Multi-component suffixes handled:
            ``'r_s'`` matches both ``r_s`` and ``r_s_c0``.
        backend : {'corner', 'dynesty'}
            Multi-file overplotting supported only for backend='corner'.
        unit_scale : dict {param_index: scale_factor}, optional
            Multiply specific parameter columns (after filtering) before
            plotting.  Applied to frozen vline values too.
        colors : list, optional
            Per-file colors.  Cycles through tab10 palette if None.
        save_plots, output_dir, use_style, n_sigma_range, return_fig, **kwargs
            As before.
        """
        if use_style:
            setup_plot_style()

        if isinstance(filename, dict):
            # {label: path} — labels given explicitly, skip auto-resolution
            filenames   = list(filename.values())
            file_labels = list(filename.keys())
        else:
            raw_inputs = [filename] if isinstance(filename, str) else list(filename)
            resolved = [self._resolve_corner_input(r) for r in raw_inputs]
            filenames   = [r[0] for r in resolved]
            file_labels = [r[1] for r in resolved]
        multi = len(filenames) > 1

        # Load all pickles
        all_data = [self._load_corner_pickle(f) for f in filenames]
        # each: (free_names, frozen_vals, labels_auto, samples, weights, sres, results)

        # ---- Determine display_params --------------------------------
        # Union strategy: include param if free in ≥1 file (frozen files get vline).
        # params filter optionally narrows the set.
        all_free: set[str] = set()
        for free_names_i, *_ in all_data:
            all_free.update(free_names_i)
        all_frozen_names: set[str] = set()
        for _, frozen_vals_i, *_ in all_data:
            all_frozen_names.update(frozen_vals_i.keys())

        candidates = all_free | all_frozen_names
        if params is not None:
            candidates = {n for n in candidates if self._param_matches(n, params)}
            if not candidates:
                available = sorted(all_free | all_frozen_names)
                raise ValueError(
                    f"No parameters matched filter {params!r}. "
                    f"Available raw names: {available}"
                )

        # Preserve order: free params in file order, then frozen-only params
        ordered: list[str] = []
        seen: set[str] = set()
        for free_names_i, *_ in all_data:
            for n in free_names_i:
                if n in candidates and n not in seen:
                    ordered.append(n)
                    seen.add(n)
        for n in sorted(candidates - seen):
            ordered.append(n)
        display_params = ordered

        # Warn about params frozen in ALL files (vlines only, no posterior)
        only_frozen = [n for n in display_params if n not in all_free]
        if only_frozen:
            warnings.warn(
                f"Parameters {only_frozen} are frozen in ALL files — "
                "only vlines will be shown (no posterior).",
                UserWarning, stacklevel=2,
            )

        n_params = len(display_params)

        # ---- Build display labels ------------------------------------
        first_free, _, first_labels_auto = all_data[0][0], all_data[0][1], all_data[0][2]
        name_to_label = dict(zip(first_free, first_labels_auto))
        # fallback: check other files' labels for params frozen in file 0
        for free_names_i, _, labels_i, *_ in all_data[1:]:
            for n, lbl in zip(free_names_i, labels_i):
                if n not in name_to_label:
                    name_to_label[n] = lbl
        display_labels = list(labels) if labels is not None else [
            name_to_label.get(n, n) for n in display_params
        ]

        # ---- Merge name-based unit scales into col_scales ---------------
        col_scales: dict[int, float] = dict(unit_scale or {})
        if unit_scale_by_name:
            for param_name, factor in unit_scale_by_name.items():
                for col_idx, dp in enumerate(display_params):
                    base = re.sub(r'_c\d+$', '', dp)
                    if base == param_name:
                        col_scales[col_idx] = factor
                        # Update label if a known unit override exists
                        if labels is None:
                            lbl_map = _UNIT_SCALE_LABELS.get(param_name, {})
                            if factor in lbl_map:
                                display_labels[col_idx] = lbl_map[factor]

        # ---- Per-file: free-column map + frozen value map ---------------
        # free_col_maps[i]: param_raw_name -> column index in samples_i
        # per_file_frozen[i]: display col_idx -> frozen value
        free_col_maps: list[dict[str, int]] = []
        all_weights: list[np.ndarray] = []
        per_file_frozen: list[dict[int, float]] = []

        for free_names_i, frozen_vals_i, _, samples_i, weights_i, _, _ in all_data:
            free_col_map = {n: idx for idx, n in enumerate(free_names_i)}
            file_frozen: dict[int, float] = {}
            for col_idx, param_name in enumerate(display_params):
                if param_name not in free_col_map:
                    val = frozen_vals_i.get(param_name)
                    if val is not None:
                        file_frozen[col_idx] = val
            free_col_maps.append(free_col_map)
            all_weights.append(weights_i)
            per_file_frozen.append(file_frozen)

        # ---- Unit conversions ----------------------------------------
        # Applied directly to each file's free samples and frozen values
        scaled_samples: list[np.ndarray] = []
        for i, (_, _, _, samples_i, _, _, _) in enumerate(all_data):
            s = np.copy(samples_i)
            for col_disp, factor in col_scales.items():
                param_name = display_params[col_disp]
                src = free_col_maps[i].get(param_name)
                if src is not None:
                    s[:, src] = s[:, src] * factor
            scaled_samples.append(s)
        for file_frozen in per_file_frozen:
            for col_idx in list(file_frozen):
                if col_idx in col_scales:
                    file_frozen[col_idx] *= col_scales[col_idx]

        # ---- Compute span from free-param samples per display column -----
        import corner as _corner
        span = []
        for r, param_name in enumerate(display_params):
            all_vals, all_w = [], []
            for i, s in enumerate(scaled_samples):
                src = free_col_maps[i].get(param_name)
                if src is not None:
                    all_vals.append(s[:, src])
                    all_w.append(all_weights[i])
            if all_vals:
                vals_cat = np.concatenate(all_vals)
                w_cat = np.concatenate(all_w)
                w_cat = w_cat / w_cat.sum()
                e = _corner.quantile(vals_cat, [0.16, 0.50, 0.84], weights=w_cat)
                span.append((
                    e[1] - n_sigma_range * abs(e[1] - e[0]),
                    e[1] + n_sigma_range * abs(e[2] - e[1]),
                ))
            else:
                # Frozen in every file: centre vline span
                fvals = [fd[r] for fd in per_file_frozen if r in fd]
                mid = float(np.median(fvals)) if fvals else 0.0
                span.append((mid - 1.0, mid + 1.0))

        if colors is None:
            colors = _DEFAULT_COLORS

        # Evidence string (single-file only)
        evidence_str = ''
        if not multi:
            results_0 = all_data[0][6]
            logz_0 = np.asarray(all_data[0][5]['logz'])
            if 'loglnull' in results_0:
                sig = float(np.sqrt(2.0 * abs(logz_0[-1] - float(results_0['loglnull']))))
                evidence_str = f'  evidence = {sig:.1f}σ'

        # ---- Backend dispatch ----------------------------------------
        fig = None
        axes = None   # n_params × n_params object array (None where no axes exist)

        if backend == 'corner':
            # Draw order: fully-free files first, then partials sorted by
            # ascending frozen count (most free params drawn earliest so the
            # maximum number of axes are created before later files need them).
            fully_free = [i for i in range(len(filenames)) if not per_file_frozen[i]]
            partial    = sorted(
                [i for i in range(len(filenames)) if per_file_frozen[i]],
                key=lambda i: len(per_file_frozen[i]),
            )
            draw_order = fully_free + partial

            for i in draw_order:
                weights_i  = all_weights[i]
                s_i        = scaled_samples[i]
                file_frozen = per_file_frozen[i]
                color_i    = colors[i % len(colors)]
                fcm        = free_col_maps[i]

                # Free-param indices within display_params for this file
                free_disp_idx   = [j for j in range(n_params) if j not in file_frozen]
                free_param_names = [display_params[j] for j in free_disp_idx]
                free_samp  = s_i[:, [fcm[n] for n in free_param_names]]
                free_span  = [span[j] for j in free_disp_idx]
                free_lbls  = [display_labels[j] for j in free_disp_idx]

                kw = dict(
                    bins=40, smooth=0.02,
                    show_titles=(fig is None),
                    quantiles=[0.16, 0.50, 0.84] if fig is None else [],
                    smooth1d=None, labels=free_lbls, labelpad=0.05,
                    label_kwargs={'fontsize': 12}, title_fmt='.4f',
                    range=free_span, plot_datapoints=False,
                    color=color_i,
                )

                if fig is None:
                    # First file creates the figure
                    kw.update(kwargs)
                    fig = _corner.corner(free_samp, weights=weights_i, **kw)
                    n_free = len(free_disp_idx)
                    tmp = np.array(fig.axes).reshape(n_free, n_free)
                    # Map into full n_params × n_params grid (None where frozen)
                    axes = np.full((n_params, n_params), None, dtype=object)
                    for ki, gi in enumerate(free_disp_idx):
                        for kj, gj in enumerate(free_disp_idx):
                            axes[gi, gj] = tmp[ki, kj]
                else:
                    # Proxy exposes only the K×K free-param subgrid to corner,
                    # so corner draws 2D contours on those axes in-place.
                    # If a required axis was never created (param free in this
                    # file but frozen in all earlier ones), insert a tiny
                    # invisible placeholder so corner doesn't crash on None.
                    free_axes_flat = []
                    for gi in free_disp_idx:
                        for gj in free_disp_idx:
                            ax_ij = axes[gi, gj]
                            if ax_ij is None:
                                ax_ij = fig.add_axes(
                                    [0, 0, 0.001, 0.001],
                                    label=f'_ph_{gi}_{gj}',
                                )
                                ax_ij.set_visible(False)
                                axes[gi, gj] = ax_ij
                            free_axes_flat.append(ax_ij)
                    kw.update(kwargs)
                    kw['fig'] = _FigProxy(fig, free_axes_flat)
                    _corner.corner(free_samp, weights=weights_i, **kw)

                # Vlines for frozen params on diagonal panels
                for col_idx, frozen_val in file_frozen.items():
                    ax = axes[col_idx, col_idx] if axes is not None else None
                    if ax is not None:
                        ax.axvline(frozen_val, color=color_i, linestyle='--',
                                   linewidth=1.5, alpha=0.8)

        elif backend == 'dynesty':
            try:
                import dynesty.plotting as dyplot
            except ImportError:
                raise ImportError("dynesty package required for backend='dynesty'")
            if multi:
                warnings.warn(
                    "Multi-file overplotting not supported for backend='dynesty'. "
                    "Only the first file will be plotted.",
                    UserWarning, stacklevel=2,
                )
            free_names_0 = all_data[0][0]
            sres_0 = all_data[0][5]
            free_disp_0 = [n for n in display_params if n in free_names_0]
            col_idx_0 = [free_names_0.index(n) for n in free_disp_0]
            mod_sres = dict(sres_0)
            mod_sres['samples'] = sres_0['samples'][:, col_idx_0]
            kw = dict(
                show_titles=True, title_fmt='.4f', max_n_ticks=5,
                labels=[display_labels[display_params.index(n)] for n in free_disp_0],
                quantiles=[0.16, 0.50, 0.84],
                label_kwargs={'fontsize': 12},
                span=[span[display_params.index(n)] for n in free_disp_0],
                color=colors[0],
            )
            kw.update(kwargs)
            fig, axes = dyplot.cornerplot(mod_sres, **kw)

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'corner' or 'dynesty'.")

        # ---- Title & legend -----------------------------------------
        if multi:
            title = ' vs '.join(file_labels)
        else:
            title = file_labels[0] + evidence_str
        fig.suptitle(title, fontsize=9, y=1.01)

        if multi:
            handles = [
                plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2,
                           label=file_labels[i])
                for i in range(len(filenames))
            ]
            fig.legend(handles=handles, loc='upper right',
                       bbox_to_anchor=(1.0, 1.0), fontsize=8)

        # ---- Save / return ------------------------------------------
        if save_plots:
            _safe_target = str(getattr(self, 'target', None) or 'unknown').replace(' ', '_')
            _prefix = f"{_safe_target}_" if getattr(self, 'target', None) else ''
            if output_dir is None:
                output_dir = f'../plots/VisualizeEszee/{_safe_target}/corner/'
            os.makedirs(output_dir, exist_ok=True)
            base = ('_vs_'.join(os.path.basename(f) for f in filenames)
                    if multi else os.path.basename(filenames[0]))
            out_path = os.path.join(output_dir, _prefix + base + '_corner.pdf')
            fig.savefig(out_path, bbox_inches='tight')
            print(f"Saved: {out_path}")

        if return_fig:
            return fig, axes
        plt.show()
