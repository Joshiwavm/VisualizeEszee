"""Corner plot mixin for dynesty nested-sampling pickle files.

Supports two backends selectable via the `backend` argument:
  - 'corner'   : uses the corner package (corner.corner)
  - 'dynesty'  : uses dynesty.plotting.cornerplot (no extra dependency)

Auto-labels are built from the pickle's vary structure + brightness_models.yml
parameter ordering, so you rarely need to pass labels manually.
"""
from __future__ import annotations

import os
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from ..utils.style import setup_plot_style

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


class PlotCorner:
    """Mixin providing plot_corner() for Manager."""

    # ------------------------------------------------------------------
    # Auto-label builder
    # ------------------------------------------------------------------
    def _get_corner_labels(self, results) -> list[str]:
        """Derive parameter labels from vary structure + YAML ordering."""
        labels: list[str] = []
        vary_list = list(results['vary'][:-1])   # exclude trailing calibration entry
        n_compts = len(vary_list)

        for j, vary in enumerate(vary_list):
            model_type = vary['values']['model'].get('type', '')
            suffix = f' (c{j})' if n_compts > 1 else ''

            # Identify parameter names for this component
            if model_type == 'pointSource':
                param_names = _PS_PARAM_NAMES
            else:
                param_names = list(self.get_param_order_from_yaml(model_type))

            # Model free params
            model_vary_flags = vary['values']['model'].get('vary', [])
            for idx, is_varied in enumerate(model_vary_flags):
                if is_varied:
                    raw = param_names[idx] if idx < len(param_names) else f'param_{idx}'
                    labels.append(_PARAM_LABELS.get(raw.lower(), raw) + suffix)

            # Spectrum free params
            spec_vary_flags = vary['values']['spectrum'].get('vary', [])
            for idx, is_varied in enumerate(spec_vary_flags):
                if is_varied:
                    labels.append(_PARAM_LABELS.get('ps_specidx', f'spec_{idx}') + suffix)

        # Calibration scales
        cal_flags = np.asarray(results['vary'][-1]['values'].get('vary', []), dtype=bool)
        scale_names = getattr(results, 'get', lambda k, d: d)('scale_names', None)
        for i, is_varied in enumerate(cal_flags):
            if is_varied:
                name = (scale_names[i] if scale_names and i < len(scale_names)
                        else f'$\\alpha_{{cal,{i}}}$')
                labels.append(name)

        return labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_corner(
        self,
        filename: str,
        labels: list[str] | None = None,
        backend: str = 'corner',
        unit_scale: dict[int, float] | None = None,
        save_output: str | None = None,
        use_style: bool = True,
        n_sigma_range: float = 4.0,
        return_fig: bool = False,
        **kwargs,
    ):
        """Make a weighted corner plot from a dynesty pickle file.

        Parameters
        ----------
        filename : str
            Path to the dynesty `.npz` pickle produced by eszee.
        labels : list of str, optional
            Axis labels. Auto-generated from vary structure if None.
        backend : {'corner', 'dynesty'}
            'corner'  — uses ``corner.corner`` (pip install corner).
            'dynesty' — uses ``dynesty.plotting.cornerplot`` (no extra dep).
        unit_scale : dict {param_index: scale_factor}, optional
            Multiply specific parameter columns before plotting.
            E.g. ``{3: 3600}`` converts column 3 from degrees to arcsec.
        save_output : str, optional
            Directory to save the PDF. Filename derived from pickle basename.
        use_style : bool
            Apply the thesis mplstyle if available.
        n_sigma_range : float
            Plot range = median ± n_sigma × half-width at [0.16, 0.84].
        return_fig : bool
            Return ``(fig, axes)`` instead of calling ``plt.show()``.
        **kwargs
            Passed through to the backend plotting function.
        """
        if use_style:
            setup_plot_style()

        results = np.load(filename, allow_pickle=True)
        sres = results['samples']
        samples = np.copy(np.asarray(sres['samples']))
        logwt = np.asarray(sres['logwt'])
        logz = np.asarray(sres['logz'])
        weights = np.exp(logwt - scipy.special.logsumexp(logwt - logz[-1]) - logz[-1])

        # Auto-label
        if labels is None:
            labels = self._get_corner_labels(results)

        # Trim to number of labels (guards against calibration rows already in samples)
        n_params = len(labels)
        samples = samples[:, :n_params]

        # Unit conversions
        if unit_scale:
            for col, factor in unit_scale.items():
                samples[:, col] = samples[:, col] * factor

        # Evidence
        evidence_str = ''
        if 'loglnull' in results:
            sig = float(np.sqrt(2.0 * abs(logz[-1] - float(results['loglnull']))))
            evidence_str = f'  evidence = {sig:.1f}σ'

        # Compute span (median ± n_sigma × half-width)
        import corner as _corner
        edges = np.array([
            _corner.quantile(samples[:, r], [0.16, 0.50, 0.84], weights=weights)
            for r in range(samples.shape[1])
        ])
        span = [
            (e[1] - n_sigma_range * abs(e[1] - e[0]),
             e[1] + n_sigma_range * abs(e[2] - e[1]))
            for e in edges
        ]

        # --- backend dispatch ---
        if backend == 'corner':
            kw = dict(
                bins=40, smooth=0.02,
                show_titles=True, quantiles=[0.16, 0.50, 0.84],
                smooth1d=None, labels=labels, labelpad=0.05,
                label_kwargs={'fontsize': 12}, title_fmt='.4f',
                range=span, plot_datapoints=False,
            )
            kw.update(kwargs)
            fig = _corner.corner(samples, weights=weights, **kw)
            axes = np.array(fig.axes).reshape(n_params, n_params)

        elif backend == 'dynesty':
            try:
                import dynesty.plotting as dyplot
            except ImportError:
                raise ImportError("dynesty package required for backend='dynesty'")
            kw = dict(
                show_titles=True, title_fmt='.4f', max_n_ticks=5,
                labels=labels, quantiles=[0.16, 0.50, 0.84],
                label_kwargs={'fontsize': 12}, span=span,
            )
            kw.update(kwargs)
            fig, axes = dyplot.cornerplot(sres, **kw)

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'corner' or 'dynesty'.")

        title = os.path.basename(filename) + evidence_str
        fig.suptitle(title, fontsize=9, y=1.01)

        if save_output is not None:
            os.makedirs(save_output, exist_ok=True)
            out_path = os.path.join(
                save_output, os.path.basename(filename) + '_corner.pdf'
            )
            fig.savefig(out_path, bbox_inches='tight')
            print(f"Saved: {out_path}")

        if return_fig:
            return fig, axes
        plt.show()
