from ..loading import Loader
from ..plot import PlotGatherer
from ..fourier import FourierManager, Deconvolve
from ..utils.utils import JyBeamToJyPix, smooth, extract_plane as utils_extract_plane, get_map_beam_and_pix as utils_get_map_beam_and_pix
import warnings
import os
import numpy as np
from astropy.io import fits

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
        self.models= {}
        self.matched_models = {}

        # Initialise loader which sets up data and model handlers
        Loader.__init__(self)

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
        else:
            recurse(target, '', True, 1)