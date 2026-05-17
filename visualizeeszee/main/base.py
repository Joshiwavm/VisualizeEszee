import re as _re

from ..loading import Loader
from ..plot import PlotGatherer
from ..fourier import FourierManager, Deconvolve
from ..utils.utils import JyBeamToJyPix, smooth, extract_plane as utils_extract_plane, get_map_beam_and_pix as utils_get_map_beam_and_pix
import warnings
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as _u, constants as _const

class Manager(Loader, FourierManager, Deconvolve, PlotGatherer):
    """
    Main manager class for loading, processing, and plotting ALMA uv-data and models.
    Unified containers:
      - self.data: {'uv': {...}, 'act': {...}}
      - self.models: model parameter / metadata records
      - self.matched_models: model->data level products (maps + sampled_model)
    Legacy accessors (properties) provided: uvdata, actdata, model_maps, sampled_model_vis.
    """
    def __init__(self, target=None):
        self.target = target
        self.data = {'uv': {}, 'act': {}}
        self.models = {}
        self.matched_models = {}
        self.point_sources = []   # populated by get_point_sources_from_pickle()
        self._ps_corrected_uvdata: set = set()  # (data_name,) keys already PS-subtracted in uvdata

        # Initialise loader which sets up data and model handlers
        Loader.__init__(self)

    # ------------------------------------------------------------------
    # Model name resolution
    # ------------------------------------------------------------------
    def _resolve_model_name(self, name: str) -> str:
        """Resolve a base model name to the nearest-q0.5 quantile key.

        Checks matched_models first, then models (for pre-match calls).
        If name already exists exactly in either pool, returns it unchanged.
        Otherwise finds all keys of the form '{name}_q{val}' and returns
        the one with val closest to 0.5.
        """
        if not name:
            return name
        for pool in (self.matched_models, self.models):
            if name in pool:
                return name
            candidates = []
            for key in pool:
                m = _re.match(rf'^{_re.escape(name)}_q([\d.]+)$', key)
                if m:
                    candidates.append((abs(float(m.group(1)) - 0.5), key))
            if candidates:
                return min(candidates)[1]
        return name

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------
    @property
    def uvdata(self):  # legacy name
        return self.data['uv']

    @property
    def actdata(self):  # legacy name
        return self.data['act']

    @property
    def model_maps(self):  # constructed view
        view = {}
        for m, dct in self.matched_models.items():
            for d, rec in dct.items():
                maps = rec.get('maps')
                if maps:
                    view.setdefault(m, {})[d] = maps
        return view

    @property
    def sampled_model_vis(self):  # compatibility alias
        view = {}
        for m, dct in self.matched_models.items():
            for d, rec in dct.items():
                sm = rec.get('sampled_model')
                if sm:
                    view.setdefault(m, {})[d] = sm
        return view
        
    # ------------------------------------------------------------------
    # Parameter summary
    # ------------------------------------------------------------------
    def summary(self, model_name: str | None = None, max_logL: bool = False):
        """Print a human-readable overview of loaded data and model parameters.

        Parameters
        ----------
        model_name : str or None
            If given, show detailed parameters only for models whose name
            starts with ``model_name``.  If None, show all models with their
            parameters.
        max_logL : bool
            If True, display MAP (peak log-likelihood) parameters for
            pickle-sourced models instead of the stored quantile values.
        """
        T, L = '├── ', '└── '

        def _fmt(key, val):
            """Format a parameter value with sensible precision."""
            if not isinstance(val, (int, float)):
                return str(val)
            if key in ('ra', 'dec'):
                return f"{val:.6f} deg"
            if key == 'redshift':
                return f"{val:.4f}"
            if key in ('mass',):
                return f"{val:.3e}"
            if key == 'p_norm':
                return f"{val:.4e}"
            if key == 'r_s':
                return f"{val:.5g} deg  ({val*3600:.2f}\")"
            return f"{val:.4g}"

        def _a10_gnfw_derived(mparams):
            """Return (p0_kevcm3, r_s_mpc, r_s_deg) derived via the A10pars transform."""
            try:
                M500 = float(mparams.get('mass', 0.0))
                c500 = float(mparams.get('c500', 1.0))
                z    = float(mparams.get('redshift', 0.0))
                bias = float(mparams.get('bias', 0.0))
                P0   = float(mparams.get('p_norm', 0.0))
                fb, mu, mue = 0.175, 0.590, 1.140
                cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
                Hz       = cosmo.H(z)
                rho_crit = cosmo.critical_density(z)
                D_A      = cosmo.angular_diameter_distance(z).to(_u.Mpc).value
                r500 = ((3.0 / 4.0 / np.pi / 500.0 / rho_crit)
                        * (1.0 - bias) * M500 * _u.solMass) ** (1.0 / 3.0)
                r_s_mpc = r500.to(_u.Mpc).value / c500
                r_s_deg = np.degrees(r_s_mpc / D_A)
                m500_log = np.log10(M500)
                p0 = P0 * (3.0 / 8.0 / np.pi) * (fb * mu / mue)
                p0 *= ((((2.5e2 * Hz * Hz) ** 2.0
                         * (1.0 - bias) * (10 ** (m500_log - 15.0)) * _u.solMass
                         / _const.G ** 0.5) ** (2.0 / 3.0)).to(_u.keV / _u.cm**3)).value
                p0 *= 1e10
                return p0, r_s_mpc, r_s_deg
            except Exception:
                return None, None, None

        def _print_params(params, indent):
            mparams = params.get('model', {})
            sparams = params.get('spectrum', {})
            mtype   = mparams.get('type', '')
            keys = [k for k in mparams if k != 'type']
            for i, k in enumerate(keys):
                pfx = L if (i == len(keys) - 1 and not sparams.get('type') and mtype != 'A10Pressure') else T
                print(f"{indent}{pfx}{k}: {_fmt(k, mparams[k])}")
            stype = sparams.get('type', '')
            if stype:
                print(f"{indent}{L}spectrum: {stype}")
            if mtype == 'A10Pressure':
                p0, r_s_mpc, r_s_deg = _a10_gnfw_derived(mparams)
                if p0 is not None:
                    print(f"{indent}{T}[gNFW equiv.]")
                    print(f"{indent}{T}  p0:  {p0:.4e} keV/cm³")
                    print(f"{indent}{L}  r_s: {r_s_mpc:.4f} Mpc  ({r_s_deg*3600:.2f}\")")

        # ── Data ──────────────────────────────────────────────────────
        uv_names  = [k for k in self.data['uv'] if k != 'metadata']
        act_names = list(self.data['act'].keys())
        print(f"Manager  target={self.target or '—'}")
        print(f"{'─'*54}")
        print(f"Data  ({len(uv_names)} interferometer, {len(act_names)} ACT):")
        all_data = uv_names + [f"{n} [ACT]" for n in act_names]
        for i, dn in enumerate(all_data):
            print(f"  {'└── ' if i == len(all_data)-1 else '├── '}{dn}")

        # ── Models ────────────────────────────────────────────────────
        print(f"\nModels  ({len(self.models)}):")
        names = list(self.models.keys())
        for mi, mn in enumerate(names):
            info   = self.models[mn]
            is_last = mi == len(names) - 1
            mtype  = info.get('type', '?')
            src    = info.get('source', '?')
            q      = info.get('quantile')
            tags   = []
            if max_logL and info.get('source') == 'pickle':
                tags.append('MAP')
            elif q is not None:
                tags.append(f"q={q}")
            if mn in self.matched_models:
                tags.append('matched')
            tag_str = f"  [{', '.join(tags)}]" if tags else ''
            print(f"  {'└── ' if is_last else '├── '}{mn}  [{mtype} | {src}]{tag_str}")

            show = (model_name is None or mn == model_name or mn.startswith(model_name))
            if show:
                params = info.get('parameters')
                # Override with MAP params if requested and model came from a pickle
                if max_logL and info.get('source') == 'pickle' and info.get('filename'):
                    try:
                        map_params, _ = self.get_parameters_from_map(info['filename'])
                        # map_params is list-of-lists (n_quants=1, n_compts)
                        compts = map_params[0]
                        params = compts[0] if len(compts) == 1 else compts
                    except Exception as e:
                        print(f"  [MAP load failed for {mn}: {e}]")
                cont = '    ' if is_last else '│   '
                if isinstance(params, list):
                    for ci, cp in enumerate(params):
                        clast = (ci == len(params) - 1)
                        print(f"  {cont}{'└── ' if clast else '├── '}c{ci}  [{cp['model'].get('type','?')}]")
                        _print_params(cp, indent=f"  {cont}{'    ' if clast else '│   '}")
                elif isinstance(params, dict) and 'model' in params:
                    _print_params(params, indent=f"  {cont}")

        # ── Point sources (from get_point_sources_from_pickle) ────────
        if self.point_sources:
            print(f"\nPoint sources  ({len(self.point_sources)}, UV-plane only):")
            for i, ps in enumerate(self.point_sources):
                pfx = '└── ' if i == len(self.point_sources) - 1 else '├── '
                if ps.get('spec_type') == 'doublePowerLaw':
                    a1_ujy = ps.get('amp1', ps['amplitude']) * 1e6
                    a2_ujy = ps.get('amp2', ps['amplitude']) * 1e6
                    print(f"  {pfx}PS{i}  ra={ps['ra']:.6f}  dec={ps['dec']:.6f}"
                          f"  [doublePowerLaw]"
                          f"  α1={ps['spec_index']:.3g}  amp1={a1_ujy:.3g} µJy @ {ps['ref_freq']/1e9:.0f} GHz"
                          f"  α2={ps.get('spec_index2', 0):.3g}  amp2={a2_ujy:.3g} µJy @ {ps.get('ref_freq2', 4e10)/1e9:.0f} GHz")
                else:
                    amp_ujy = ps['amplitude'] * 1e6
                    print(f"  {pfx}PS{i}  ra={ps['ra']:.6f}  dec={ps['dec']:.6f}"
                          f"  amp={amp_ujy:.3g} µJy  α={ps['spec_index']:.3g}"
                          f"  ref_freq={ps['ref_freq']/1e9:.0f} GHz")

        # ── Matched summary ───────────────────────────────────────────
        if self.matched_models:
            print(f"\nMatched  ({len(self.matched_models)}):")
            mm = list(self.matched_models.items())
            for i, (mn, ddict) in enumerate(mm):
                data_keys = [k for k in ddict if k not in ('status', 'notes')]
                print(f"  {'└── ' if i == len(mm)-1 else '├── '}{mn}  →  {', '.join(data_keys)}")

    # ------------------------------------------------------------------
    # Inspection helper
    # ------------------------------------------------------------------
    def dump_structure(self, model_name: str | None = None, data_name: str | None = None,
                       *, depth: int | None = None, summarize_arrays: bool = True,
                       max_list: int = 5):
        """Print nested keys as an ASCII tree.

        Parameters
        ----------
        model_name, data_name : optional filters
        depth : int or None
            Maximum depth to descend (None = unlimited).
        summarize_arrays : bool
            If True, show ndarray shape instead of full content.
        max_list : int
            Max elements to preview for list/tuple.
        """
        if model_name is not None:
            model_name = self._resolve_model_name(model_name)
        target = self.matched_models
        if model_name is not None:
            target = target.get(model_name, {})
        if data_name is not None and isinstance(target, dict):
            target = target.get(data_name, {})

        def short_value(v):
            import numpy as _np
            if summarize_arrays and isinstance(v, _np.ndarray):
                return f"ndarray shape={v.shape}" if v.ndim else f"ndarray len={len(v)}"
            if isinstance(v, (list, tuple)):
                show = v[:max_list]
                more = '...' if len(v) > max_list else ''
                return f"[{', '.join(map(str, show))}{more}]"
            if isinstance(v, (int, float, str)):
                s = str(v)
                return s if len(s) < 40 else s[:37] + '...'
            return type(v).__name__

        lines = []
        def recurse(obj, prefix: str, is_last: bool, level: int):
            if depth is not None and level > depth:
                return
            if isinstance(obj, dict):
                keys = list(obj.keys())
                for i, k in enumerate(keys):
                    v = obj[k]
                    last = (i == len(keys)-1)
                    branch = '`-' if last else '|-'
                    if isinstance(v, dict):
                        lines.append(f"{prefix}{branch} {k}")
                        extend = '  ' if last else '| '
                        recurse(v, prefix + extend, last, level+1)
                    else:
                        val_repr = short_value(v)
                        lines.append(f"{prefix}{branch} {k}: {val_repr}")
            else:
                lines.append(f"{prefix}`- {short_value(obj)}")

        # Root handling
        if isinstance(target, dict):
            if not target:
                print('(empty)')
                return
            # If both model and data specified, start at that node without extra root label
            recurse(target, '', True, 1)
            # print assembled lines
            for ln in lines:
                print(ln)
        else:
            recurse(target, '', True, 1)
            for ln in lines:
                print(ln)