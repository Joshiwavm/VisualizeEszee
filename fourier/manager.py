from __future__ import annotations
from typing import Tuple, Sequence, Dict, Any, Optional

import numpy as np
from scipy.interpolate import RectBivariateSpline

import jax  # noqa: F401
import jax_finufft  # noqa: F401

from ..utils.utils import (
    ytszToJyPix,
    JyBeamToJyPix,  # retained for potential external use, not applied in Jy/pix path
    comptonCorrect,
    computeFlatCompton,
    comptonRelativ,
)

class FourierManager:
    """Manage Fourier-domain operations and spectral scaling.

    Attributes
    ----------
    models : dict
        Registry of model definitions & parameters (expected external assignment).
    matched_models : dict
        Nested structure linking models to datasets and their map entries.
    uvdata : dict
        Observational uv datasets keyed by data / field / spw.
    """

    # ------------------------------------------------------------------
    # Spectral scaling helpers (non-tSZ)
    # ------------------------------------------------------------------
    @staticmethod
    def _scale_powerlaw(spec: Dict[str, Any], freq: float) -> float:
        ref = float(spec.get('nu0', freq) or freq)
        idx = float(spec.get('SpecIndex', spec.get('specindex', 0.0)))
        amp = float(spec.get('amp', 1.0))
        x = freq / ref if ref else 1.0
        return amp * (x ** idx)

    @staticmethod
    def _scale_powerlawmod(spec: Dict[str, Any], freq: float) -> float:
        ref = float(spec.get('nu0', freq) or freq)
        idx = float(spec.get('SpecIndex', spec.get('specindex', 0.0)))
        curv = float(spec.get('SpecCurv', spec.get('speccurv', 0.0)))
        amp = float(spec.get('amp', 1.0))
        x = freq / ref if ref else 1.0
        idx_eff = idx + curv * np.log(max(x, 1e-30))
        return amp * (x ** idx_eff)

    @staticmethod
    def _scale_powerdust(spec: Dict[str, Any], freq: float) -> float:
        ref = float(spec.get('nu0', freq) or freq)
        idx = float(spec.get('SpecIndex', spec.get('specindex', 0.0)))
        beta = float(spec.get('beta', 0.0))
        curv = float(spec.get('SpecCurv', spec.get('speccurv', 0.0)))
        amp = float(spec.get('amp', 1.0))
        x = freq / ref if ref else 1.0
        base = idx + beta + curv * np.log(max(x, 1e-30))
        return amp * (x ** base)

    @staticmethod
    def _scale_scaling(spec: Dict[str, Any], freq: float) -> float:
        return float(spec.get('amp', 1.0))

    def _spectral_scale(self, spec: Dict[str, Any], freq: float) -> float:
        stype = str(spec.get('type', '')).lower()
        if stype == 'powerlaw':
            return self._scale_powerlaw(spec, freq)
        if stype == 'powerlawmod':
            return self._scale_powerlawmod(spec, freq)
        if stype == 'powerdust':
            return self._scale_powerdust(spec, freq)
        if stype == 'scaling':
            return self._scale_scaling(spec, freq)
        return float(spec.get('amp', 1.0))

    @staticmethod
    def map_to_uvgrid(image: np.ndarray, pixel_scale_deg: float) -> Tuple[np.ndarray, float]:
        # Flip vertical axis (legacy convention) then FFT to uv half-plane
        img = np.ascontiguousarray(np.flip(image, axis=0))
        N = img.shape[0]
        delt_rad = np.deg2rad(abs(pixel_scale_deg))
        du = 1.0 / (N * delt_rad)
        uv_rfft_shifted = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(img)), axes=0)
        return uv_rfft_shifted, du

    @staticmethod
    def sample_uv(uv_rfft_shifted: np.ndarray, u: Sequence[float], v: Sequence[float], du: float,
                  dRA: float = 0.0, dDec: float = 0.0, PA: float = 0.0, origin: str = 'upper') -> np.ndarray:
        """Interpolate uv grid at (u,v) with rotation & phase offsets."""
        # u = np.asarray(u); v = np.asarray(v)
        
        v_origin = 1.0 if origin == 'upper' else -1.0
        nxy = uv_rfft_shifted.shape[0]
        
        cos_PA = np.cos(PA); sin_PA = np.sin(PA)

        urot = u * cos_PA - v * sin_PA
        vrot = u * sin_PA + v * cos_PA

        dRArot = dRA * cos_PA - dDec * sin_PA
        dDecrot = dRA * sin_PA + dDec * cos_PA

        uroti = np.abs(urot) / du
        vroti = nxy/2.0 + v_origin * vrot / du
        uneg = urot < 0.0
        vroti[uneg] = nxy/2.0 - v_origin * vrot[uneg]/du

        u_axis = np.linspace(0.0, nxy//2, nxy//2 + 1)
        v_axis = np.linspace(0.0, nxy - 1, nxy)
        
        f_re = RectBivariateSpline(v_axis, u_axis, uv_rfft_shifted.real, kx=1, ky=1, s=0)
        f_im = RectBivariateSpline(v_axis, u_axis, uv_rfft_shifted.imag, kx=1, ky=1, s=0)
        f_amp = RectBivariateSpline(v_axis, u_axis, np.abs(uv_rfft_shifted), kx=1, ky=1, s=0)
        ReInt = f_re.ev(vroti, uroti)
        ImInt = f_im.ev(vroti, uroti)
        AmpInt = f_amp.ev(vroti, uroti)
        ImInt[uneg] *= -1.0
        PhaseInt = np.angle(ReInt + 1j * ImInt)
        theta = urot * dRArot + vrot * dDecrot

        return AmpInt * (np.cos(theta + PhaseInt) + 1j * np.sin(theta + PhaseInt))

    @staticmethod
    def vis_to_image(u: Sequence[float], v: Sequence[float], vis: Sequence[complex],
                     weights: Optional[Sequence[float]] = None,
                     npix: int | None = None,
                     pixel_scale_deg: float | None = None,
                     normalize: bool = True) -> np.ndarray:
        """Dirty image reconstruction via NUFFT (requires jax + jax_finufft)."""
        u = np.asarray(u); v = np.asarray(v); vis = np.asarray(vis)
        delt_rad = np.deg2rad(abs(pixel_scale_deg))
        x = -2.0 * np.pi * v * delt_rad
        y =  2.0 * np.pi * u * delt_rad
        wsum = weights.sum()
        if wsum == 0:
            return np.zeros((npix, npix))
        coeffs = weights * vis
        if normalize:
            coeffs = coeffs / wsum
        grid = jax_finufft.nufft1((npix, npix), coeffs.astype(np.complex128), x, y)
        return np.array(grid.real)

    def _multivis_fields_to_image(self, fields: Sequence[Dict[str, Any]], npix: int, pixel_scale_deg: float,
                                  align: bool = True, normalize: bool = True) -> np.ndarray:
        """Combine multiple fields (optionally phase-align) into a single dirty image.

        This is a low-level helper that accepts an iterable of field dicts with keys
        'u','v','vis' and optional 'weights','dRA','dDec'. Use the instance method
        `multivis_to_image` to build these field dicts from sampled_model entries.
        """
        accum_u, accum_v, accum_vis, accum_w = [], [], [], []
        for fd in fields:

            print(fd['name'])
            u = np.asarray(fd['u']); v = np.asarray(fd['v'])
            w = np.asarray(fd['weights']); vis = np.asarray(fd['vis']) 

            if align:
                dRA = fd.get('dRA', 0.0); dDec = fd.get('dDec', 0.0)
                vis = self.phase_shift(vis, u, v, dRA=dRA, dDec=dDec)
            
            accum_u.append(u); accum_v.append(v); accum_vis.append(vis); accum_w.append(w)
        
        u_all = np.concatenate(accum_u); v_all = np.concatenate(accum_v)
        vis_all = np.concatenate(accum_vis); w_all = np.concatenate(accum_w)
        return FourierManager.vis_to_image(u_all, v_all, vis_all, weights=w_all,
                                           npix=npix, pixel_scale_deg=pixel_scale_deg,
                                           normalize=normalize)

    def multivis_to_image(self, model_name: str, data_name: str,
                          use: str = 'data', 
                          fields: Optional[Sequence[str]] = None,
                          spws: Optional[Sequence[str]] = None,
                          npix: int = 1024, pixel_scale_deg: Optional[float] = None,
                          normalize: bool = True, calib: float = 1.0,
                          align: bool = True) -> np.ndarray:
        """Build a dirty image from already-sampled visibilities for a model/data pair.

        Parameters
        ----------
        model_name, data_name : str
            Keys into `self.matched_models` and `self.uvdata`.
        use : {'data','model','resid'}
            Which visibilities to image.
        fields : sequence[str] or None
            Subset of field keys to include (None => all sampled fields).
        spws : sequence[str] or None
            Subset of spws to include (None => all sampled spws).
        npix : int or None
            Output image size in pixels. If None, inferred from model maps (smallest pixel-scale choice).
        pixel_scale_deg : float or None
            Pixel scale in degrees. If None, choose smallest pixel scale among model maps.
        normalize, calib, align : see documentation

        Returns
        -------
        image : np.ndarray
            Dirty image (npix x npix)
        """
        # Validate presence
        if model_name not in self.matched_models:
            raise ValueError(f"Model '{model_name}' not found in matched_models")
        if data_name not in self.matched_models[model_name]:
            raise ValueError(f"Data '{data_name}' not found for model '{model_name}'")

        mm_entry = self.matched_models[model_name][data_name]

        # Ensure sampled_model exists; if not, call sample_fft for each map entry
        sampled = mm_entry.get('sampled_model', {})
        if not sampled:
            maps = mm_entry.get('maps', {})
            for field_key, spw_dict in maps.items():
                for spw_key in spw_dict.keys():
                    # populate sampled_model entries using provided calib
                    self.sample_fft(model_name, data_name, calib, field_key, spw_key)
            sampled = mm_entry.get('sampled_model', {})

        # Build list of field dicts for the low-level helper
        field_dicts = []
        pixel_scales = []
        pix_N_map = []

        # Determine central phase center for optional alignment
        central_field = self.find_central_field(data_name)
        central_phase = self.uvdata[data_name][central_field]['phase_center']

        # Iterate sampled entries
        for field_key, spw_map in sampled.items():
            if (fields is not None) and (field_key not in fields):
                continue
            # attempt to get pixel scale info from fft_map for this field/spw
            for spw_key, entry in spw_map.items():
                if (spws is not None) and (spw_key not in spws):
                    continue

                # collect pixel-scale info directly from stored map header
                cd1 = self.matched_models[model_name][data_name]['maps'][field_key][spw_key]['header'].get('CDELT1') or \
                        self.matched_models[model_name][data_name]['maps'][field_key][spw_key]['header'].get('CD1_1')
                pixel_scales.append(abs(cd1))

                plane = self._extract_plane(self.matched_models[model_name][data_name]['maps'][field_key][spw_key].get('model_data'))
                pix_N_map.append((int(np.asarray(plane).shape[0]) if plane is not None else np.nan, cd1))

                # Choose which visibility to use
                if use == 'data':
                    vis = entry['data_vis']
                elif use == 'model':
                    vis = entry['model_vis']
                elif use == 'resid':
                    vis = entry['resid_vis']
                else:
                    raise ValueError("use must be 'data', 'model' or 'resid'")

                fd = {
                    'u': entry['u'],
                    'v': entry['v'],
                    'vis': vis,
                    'weights': entry['weights'],
                    'name': model_name + data_name + field_key + spw_key
                }

                # optional alignment offsets (degrees)
                if align and (central_phase is not None):
                    field_phase = self.uvdata[data_name].get(field_key, {}).get('phase_center')
                    dRA = np.deg2rad(central_phase[0] - field_phase[0])
                    dDec = np.deg2rad(central_phase[1] - field_phase[1])
                    fd['dRA'] = dRA
                    fd['dDec'] = dDec

                field_dicts.append(fd)

        # Choose pixel_scale_deg: smallest if mixed
        if pixel_scale_deg is None and pixel_scales:
            pixel_scale_deg = float(np.nanmin(pixel_scales))

        # Call the low-level helper to build the image
        image = self._multivis_fields_to_image(field_dicts, npix=npix,
                                                pixel_scale_deg=pixel_scale_deg,
                                                align=align, normalize=normalize)
        return image



    @staticmethod
    def residual_vis(data_vis: Sequence[complex], model_vis: Sequence[complex]) -> np.ndarray:
        """Compute residual visibilities (data - model)."""
        data_vis = np.asarray(data_vis); model_vis = np.asarray(model_vis)
        if data_vis.shape != model_vis.shape:
            raise ValueError("data_vis and model_vis shape mismatch")
        return data_vis - model_vis

    @staticmethod
    def scale_vis(vis: Sequence[complex], factor: float) -> np.ndarray:
        """Scalar multiply visibility array."""
        return np.asarray(vis) * factor

    @staticmethod
    def phase_shift(vis: Sequence[complex], u: Sequence[float], v: Sequence[float],
                    dRA: float = 0.0, dDec: float = 0.0) -> np.ndarray:
        """Apply phase shift corresponding to RA/Dec offsets (radians).
        """
        vis = np.asarray(vis); u = np.asarray(u); v = np.asarray(v)
        if vis.shape != u.shape:
            raise ValueError("vis and u/v shapes mismatch")
        phase = np.exp(-2j * np.pi * (u * dRA + v * dDec))
        return vis * phase

    @staticmethod
    def _extract_plane(arr):
        """Return 2D plane from 2D/3D/4D (stokes/freq) array layouts."""
        a = np.asarray(arr)
        if a.ndim == 4:
            return a[0, 0]
        if a.ndim == 3:
            return a[0]
        return a

    def _convert_model_map_to_jypix(self, model_name, entry):
        """Convert model image to Jy/pixel (no beam normalization).

        tSZ: y * dI/dy (including relativistic correction).
        non-tSZ: apply analytic spectral scale factor (assumed already Jy-like per pixel).
        Returns (image_JyPerPixel, pixel_scale_deg).
        """
        header = entry['header']
        model_plane = self._extract_plane(entry['model_data'])
        model_plane = np.nan_to_num(model_plane, nan=0.0, posinf=0.0, neginf=0.0)

        # Frequency & pixel scale --------------------------------------------------
        # Frequency: rely solely on CRVAL3 (ignore potentially incorrect RESTFRQ)
        freq = header.get('CRVAL3')
        if freq is None:
            raise ValueError("Frequency not found in header (CRVAL3)")
        cd1 = header.get('CDELT1') or header.get('CD1_1')
        cd2 = header.get('CDELT2') or header.get('CD2_2')
        if cd1 is None or cd2 is None:
            raise ValueError("Pixel scale missing (CDELT1/2 or CD*_*).")
        ipix_deg, jpix_deg = abs(cd1), abs(cd2)

        # Spectrum type (standardized structure) ----------------------------------
        spec_type = self.models[model_name]['parameters']['spectrum']['type']
        tsz_bool = isinstance(spec_type, str) and spec_type.lower() == 'tsz'

        if not tsz_bool:
            spec = self.models[model_name]['parameters']['spectrum']
            scale = self._spectral_scale(spec, freq)
            return model_plane * scale, ipix_deg

        # tSZ conversion path ------------------------------------------------------
        # Temperature extraction (optional) ---------------------------------------
        Te = self.models['0459_1']['parameters']['model']['temperature']

        osz = 4
        # Compute relativistic series coefficients (flat band; single freq treated as degenerate interval)
        # Legacy does band-averaged comptonFlat first if starting in Jy; here we construct y->Jy/pix factor directly.
        series = np.array([
            ytszToJyPix(freq, ipix_deg, jpix_deg) * comptonRelativ(freq, order)
            for order in range(1 + osz)
        ])
        conv = comptonCorrect(series, Te=Te)  # full dI/dy including relativistic corrections
        jy_pix = model_plane * conv
        return jy_pix, ipix_deg

    # ------------------------------------------------------------------
    # New sampling API against matched_models structure
    # ------------------------------------------------------------------
    def fft_map(self, model_name, data_name, field_key, spw_key):
        """Produce uv grid (half-plane) for a model/data/spw combination."""
        img                  = self.matched_models[model_name][data_name]['maps'][field_key][spw_key]
        img_jypix, pix_deg   = self._convert_model_map_to_jypix(model_name, img)
        uv_grid, du          = self.map_to_uvgrid(img_jypix, pix_deg)
        return {'uv': uv_grid, 'du': du, 'pixscale_deg': pix_deg}

    def sample_fft(self, model_name, data_name, calib, field_key, spw_key):
        """Sample model uv grid at observed (u,v) coordinates; store results."""

        # get uvdata
        uvrec = self.uvdata[data_name][field_key][spw_key]
        u, v = uvrec.uwave, uvrec.vwave
        real, imag, wgt = uvrec.uvreal, uvrec.uvimag, uvrec.suvwght
        freq = uvrec.uvfreq  # added frequency array

        # fft map 
        uv_entry = self.fft_map(model_name, data_name, field_key, spw_key)
        model_vis = self.sample_uv(uv_entry['uv'], u, v, uv_entry['du'])

        # compute residuals
        data_vis = real + 1j * imag
        resid_vis = data_vis - model_vis * calib

        # store sampled model
        sm_store = self.matched_models[model_name][data_name].setdefault('sampled_model', {})
        sm_store.setdefault(field_key, {})[spw_key] = {
            'model_vis': model_vis * calib,
            'data_vis': data_vis,
            'resid_vis': resid_vis,
            'u': u,
            'v': v,
            'uvfreq': freq,
            'weights': wgt
        }
        return model_vis

    def map_to_vis(self, model_name, data_name, calib, field_key, spw_key):
        """Convenience wrapper returning (model_vis, resid_vis)."""
        self.sample_fft(model_name, data_name, calib, field_key, spw_key)
        entry = self.matched_models[model_name][data_name]['sampled_model'][field_key][spw_key]
        return entry['model_vis'], entry['resid_vis']