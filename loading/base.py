from .data_handler import DataHandler
from .model_handler import ModelHandler
from ..plot import PlotGatherer
from ..fourier import FourierManager
import warnings

class PlotManager(FourierManager, DataHandler, ModelHandler, PlotGatherer):
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

        DataHandler.__init__(self)  # will operate on properties
        ModelHandler.__init__(self)

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
    # Matching: always (re)build Fourier products (no flags)
    # ------------------------------------------------------------------
    def _match_single(self, model_name: str, data_name: str, notes=None):
        meta = self.uvdata[data_name].get('metadata', {})
        if meta.get('obstype','').lower() != 'interferometer':
            return None
        model_info = self.models[model_name]
        if data_name not in model_info:
            # attach spatial metadata snapshot for record
            model_info[data_name] = {
                'band': meta.get('band'),
                'array': meta.get('array'),
                'fields': meta.get('fields'),
                'spws': meta.get('spws'),
                'binvis': meta.get('binvis')
            }
            # build maps for this pair
            maps = self.add_model_maps(model_name, dataset_name=data_name)
        else:
            # ensure maps exist (rebuild each match for simplicity)
            maps = self.add_model_maps(model_name, dataset_name=data_name)
        assoc = self.matched_models.setdefault(model_name, {}).setdefault(data_name, {})
        assoc.update({'status': 'fourier_pending', 'notes': notes, 'maps': maps, 'sampled_model': {}})
        # Build Fourier products
        fields = meta.get('fields', [])
        spws_nested = meta.get('spws', [])
        for f, field in enumerate(fields):
            field_key = f'field{field}'
            for spw in spws_nested[f]:
                spw_key = f'spw{spw}'
                self.map_to_vis(model_name, data_name, field_key, spw_key)
        assoc['status'] = 'fourier_ready'

    def match_model(self, model_name: str | None = None, data_name: str | None = None, *, notes=None):
        model_list = list(self.models.keys()) if model_name is None else [model_name]
        if not model_list:
            raise ValueError("No models available to match.")
        def is_interf(d):
            return self.uvdata[d].get('metadata', {}).get('obstype','').lower() == 'interferometer'
        if data_name is None:
            data_list = [k for k in self.uvdata.keys() if is_interf(k)]
        else:
            if data_name not in self.uvdata:
                raise ValueError(f"Data set '{data_name}' not found.")
            data_list = [data_name] if is_interf(data_name) else []
        if not data_list:
            return
        for m in model_list:
            for d in data_list:
                self._match_single(m, d, notes=notes)

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
        print('\n'.join(lines))