from .data_handler import DataHandler
from .model_handler import ModelHandler
from ..plot import PlotGatherer
from ..fourier import FourierManager
from ..utils.utils import JyBeamToJyPix, smooth
import warnings
import os
import numpy as np
from astropy.io import fits

class Manager(FourierManager, DataHandler, ModelHandler, PlotGatherer):
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

    def JvM_clean(self, model_name: str | None = None, data_name: str | None = None, *, notes=None, save_output=None):
        """Perform a Jansky/Beam (JvM) style combination: add the model map (Jy/beam,
        smoothed to the smallest data beam) to the residual dirty image produced
        from sampled visibilities.

        Returns
        -------
        jvm_image : np.ndarray
            Combined residual + smoothed model image in units of Jy/beam.
        """
        if model_name is None or data_name is None:
            raise ValueError("model_name and data_name must be provided for JvM_clean")

        if model_name not in self.matched_models:
            raise ValueError(f"Model '{model_name}' not found in matched_models")
        if data_name not in self.matched_models[model_name]:
            raise ValueError(f"Data '{data_name}' not found for model '{model_name}'")

        assoc = self.matched_models[model_name][data_name]
        maps = assoc.get('maps', {})
        if not maps:
            raise ValueError("No model maps available to perform JvM_clean")

        # Find the model map with the smallest beam area (BMAJ * BMIN)
        best_entry = None
        best_field = None
        best_spw = None
        best_area = np.inf
        # Also find the smallest data beam across all maps to define target beam
        target_bmaj = None
        target_bmin = None
        target_area = np.inf

        for field_key, spw_dict in maps.items():
            for spw_key, entry in spw_dict.items():
                header = entry.get('header', {})
                bmaj = header.get('BMAJ')
                bmin = header.get('BMIN')
                if bmaj is None or bmin is None:
                    area = np.inf
                else:
                    try:
                        area = float(bmaj) * float(bmin)
                    except Exception:
                        area = np.inf

                # choose smallest-beam model map
                if area < best_area:
                    best_area = area
                    best_entry = entry
                    best_field = field_key
                    best_spw = spw_key

                # update target (smallest data beam across maps)
                if area < target_area:
                    target_area = area
                    target_bmaj = bmaj
                    target_bmin = bmin

        if best_entry is None:
            raise ValueError("Could not locate a valid model map with beam info")

        # Convert chosen model map to Jy/pixel (handles tSZ or other spectra)
        model_jypix, pix_deg = self._convert_model_map_to_jypix(model_name, best_entry)

        # Read pixel sizes (two axes) from header for accurate conversion
        header = best_entry.get('header', {})
        cd1 = header.get('CDELT1') or header.get('CD1_1')
        cd2 = header.get('CDELT2') or header.get('CD2_2')
        if cd1 is None or cd2 is None:
            raise ValueError("Pixel scale (CDELT1/CDELT2 or CD*) missing from model header")
        ipix_deg = abs(float(cd1)); jpix_deg = abs(float(cd2))

        # Convert Jy/pixel -> Jy/beam using header beam if present
        bmaj_model = header.get('BMAJ')
        bmin_model = header.get('BMIN')
        if bmaj_model is None or bmin_model is None:
            # No beam info: treat model as already Jy/beam and skip conversion
            model_jybeam = model_jypix
        else:
            # JyBeamToJyPix returns factor that converts Jy/beam -> Jy/pix,
            # so divide Jy/pix by that factor to obtain Jy/beam
            factor = JyBeamToJyPix(ipix_deg, jpix_deg, float(bmaj_model), float(bmin_model))
            if factor == 0 or np.isnan(factor):
                model_jybeam = model_jypix
            else:
                model_jybeam = model_jypix / factor

        # Determine smoothing kernel to reach the smallest data beam
        if target_bmaj is None or target_bmin is None or bmaj_model is None or bmin_model is None:
            sigma_conv = 0.0
        else:
            # Use geometric mean FWHM as equivalent circular FWHM (degrees)
            fwhm_model_deg = np.sqrt(float(bmaj_model) * float(bmin_model))
            fwhm_target_deg = np.sqrt(float(target_bmaj) * float(target_bmin))

            # Convert FWHM (deg) to pixels
            fwhm_model_pix = fwhm_model_deg / ipix_deg
            fwhm_target_pix = fwhm_target_deg / ipix_deg

            # Convert to Gaussian sigma (pixels)
            sigma_model = fwhm_model_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sigma_target = fwhm_target_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            delta2 = sigma_target**2 - sigma_model**2
            sigma_conv = np.sqrt(delta2) if delta2 > 0.0 else 0.0

        # Smooth model (Jy/beam) if required
        if sigma_conv > 0.0:
            model_smoothed = smooth(model_jybeam, sigma_conv)
        else:
            model_smoothed = model_jybeam

        # Build residual dirty image using sampled visibilities on the chosen grid
        npix = model_smoothed.shape[0]
        pixel_scale_deg = ipix_deg
        resid = self.multivis_to_image(model_name, data_name, use='resid', npix=npix, pixel_scale_deg=pixel_scale_deg, calib=1.0, align=False)

        # Resid and model are both in Jy/beam -> add
        jvm_image = resid + model_smoothed

        # Store and optionally save
        assoc.setdefault('JvM_clean', {})
        assoc['JvM_clean'][f'{best_field}_{best_spw}'] = jvm_image.astype(np.float32)

        if save_output is not None:
            os.makedirs(save_output, exist_ok=True)
            h = header.copy(); h['BUNIT'] = 'Jy/beam'
            fname = os.path.join(save_output, f"{model_name}_{data_name}_{best_field}_{best_spw}_JvM_clean.fits")
            print(f"Writing JvM-cleaned image to {fname}")
            fits.writeto(fname, jvm_image.astype(np.float32), header=h, overwrite=True)

        return jvm_image

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
        y_map = self._extract_plane(entry['model_data'])  # (y * PB)

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