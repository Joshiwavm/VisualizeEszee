from ..loading import DataHandler
from ..loading import ModelHandler
from ..plot import PlotGatherer
from ..fourier import FourierManager, Deconvolve
from ..utils.utils import JyBeamToJyPix, smooth, extract_plane as utils_extract_plane, get_map_beam_and_pix as utils_get_map_beam_and_pix
import warnings
import os
import numpy as np
from astropy.io import fits

class Manager(DataHandler, ModelHandler, FourierManager, Deconvolve, PlotGatherer):
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
    def _match_single(self, model_name: str, data_name: str, calib: float,
                      notes=None, save_output=None):
        meta = self.uvdata[data_name].get('metadata', {})
        if meta.get('obstype','').lower() != 'interferometer':
            return None
        model_info = self.models[model_name]

        model_info[data_name] = {
            'band': meta.get('band'),
            'array': meta.get('array'),
            'fields': meta.get('fields'),
            'spws': meta.get('spws'),
            'binvis': meta.get('binvis')
        }
        # build maps for this pair
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
                self.map_to_vis(model_name, data_name, calib, field_key, spw_key)
                if save_output is not None:
                    self._save_match_outputs(model_name, data_name, field_key, spw_key, assoc, save_output)
        assoc['status'] = 'fourier_ready'

    def match_model(self, model_name: str | None = None, data_name: str | None = None, *, 
                    notes=None, save_output=None, calib: float = 1.0):
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
                self._match_single(m, d, calib, notes=notes, save_output=save_output)

    # ------------------------------------------------------------------
    # Output writer helper
    # ------------------------------------------------------------------
    def _save_match_outputs(self, model_name, data_name, field_key, spw_key, assoc, save_output):
        os.makedirs(save_output, exist_ok=True)  # ensure output root exists

        # Map & vis entries
        entry = assoc['maps'][field_key][spw_key]
        sm_entry = assoc['sampled_model'][field_key][spw_key]
        header = entry['header'].copy()

        # Recover Compton-y (remove PB attenuation)
        y_map = self.__extract_plane(entry['model_data'])  # (y * PB)

        # Write Compton-y FITS
        header_y = header.copy(); header_y['BUNIT'] = 'Compton-y'
        print(f"Writing Compton-y map to {os.path.join(save_output, f'{model_name}_{data_name}_{field_key}_{spw_key}_y.fits')}")
        fits.writeto(os.path.join(save_output, f"{model_name}_{data_name}_{field_key}_{spw_key}_y.fits"), y_map.astype(np.float32), header=header_y, overwrite=True)

        # Build dirty images (model & residual) from visibilities
        uvrec = self.uvdata[data_name][field_key][spw_key]
        u = uvrec.uwave; v = uvrec.vwave; w = uvrec.suvwght
        model_vis = sm_entry['model_vis']; resid_vis = sm_entry['resid_vis']
        pix_deg = abs(header.get('CDELT1') or header.get('CD1_1'))
        npix = y_map.shape[0]
        dirty_model = self.vis_to_image(u, v, model_vis, weights=w, npix=npix, pixel_scale_deg=pix_deg, normalize=True)
        dirty_resid = self.vis_to_image(u, v, resid_vis, weights=w, npix=npix, pixel_scale_deg=pix_deg, normalize=True)

        # Write dirty images (Jy/beam)
        header_dm = header.copy(); header_dm['BUNIT'] = 'Jy/beam'
        header_dr = header.copy(); header_dr['BUNIT'] = 'Jy/beam'

        print(f"Writing dirty model to {os.path.join(save_output, f'{model_name}_{data_name}_{field_key}_{spw_key}_dirty_model.fits')}")
        print(f"Writing dirty residual to {os.path.join(save_output, f'{model_name}_{data_name}_{field_key}_{spw_key}_dirty_resid.fits')}")

        fits.writeto(os.path.join(save_output, f"{model_name}_{data_name}_{field_key}_{spw_key}_dirty_model.fits"), dirty_model.astype(np.float32), header=header_dm, overwrite=True)
        fits.writeto(os.path.join(save_output, f"{model_name}_{data_name}_{field_key}_{spw_key}_dirty_resid.fits"), dirty_resid.astype(np.float32), header=header_dr, overwrite=True)

        # Persist computed maps inside in-memory structure for easier reuse
        entry['dirty_model'] = dirty_model.astype(np.float32)
        entry['dirty_resid'] = dirty_resid.astype(np.float32)

        
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