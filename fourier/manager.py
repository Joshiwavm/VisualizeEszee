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
                     normalize: bool = True,
                     use_jax: bool = True) -> np.ndarray:
        """Dirty image reconstruction via NUFFT (requires jax + jax_finufft)."""
        if not use_jax:
            raise ValueError("vis_to_image requires JAX backend.")
        u = np.asarray(u); v = np.asarray(v); vis = np.asarray(vis)
        if u.shape != v.shape or u.shape != vis.shape:
            raise ValueError("u, v, vis must share shape")
        if npix is None or pixel_scale_deg is None:
            raise ValueError("npix and pixel_scale_deg required")
        if weights is None:
            weights = np.ones_like(u, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
        if weights.shape != u.shape:
            raise ValueError("weights shape mismatch")
        if weights.size == 0:
            return np.zeros((npix, npix))
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

    @staticmethod
    def multi_field_dirty(fields: Sequence[Dict[str, Any]], npix: int, pixel_scale_deg: float,
                          align: bool = True, normalize: bool = True) -> np.ndarray:
        """Combine multiple fields (optionally phase-align) into a single dirty image."""
        accum_u, accum_v, accum_vis, accum_w = [], [], [], []
        for fd in fields:
            u = np.asarray(fd['u']); v = np.asarray(fd['v']); vis = np.asarray(fd['vis'])
            if align and ('dRA' in fd or 'dDec' in fd):
                dRA = fd.get('dRA', 0.0); dDec = fd.get('dDec', 0.0)
                phase = np.exp(2.0 * np.pi * 1j * (u * dRA + v * dDec))
                vis *= phase
            w = np.asarray(fd.get('weights', np.ones_like(u)))
            accum_u.append(u); accum_v.append(v); accum_vis.append(vis); accum_w.append(w)
        if not accum_u:
            return np.zeros((npix, npix))
        u_all = np.concatenate(accum_u); v_all = np.concatenate(accum_v)
        vis_all = np.concatenate(accum_vis); w_all = np.concatenate(accum_w)
        return FourierManager.vis_to_image(u_all, v_all, vis_all, weights=w_all,
                                           npix=npix, pixel_scale_deg=pixel_scale_deg,
                                           normalize=normalize)

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
        """Apply phase shift corresponding to RA/Dec offsets (degrees)."""
        vis = np.asarray(vis); u = np.asarray(u); v = np.asarray(v)
        if vis.shape != u.shape:
            raise ValueError("vis and u/v shapes mismatch")
        phase = np.exp(2.0 * np.pi * 1j * (u * dRA + v * dDec))
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
        freq = None
        for k in ("RESTFRQ", "CRVAL3"):
            if k in header and header[k]:
                freq = header[k]; break
        if freq is None:
            raise ValueError("Frequency not found in header (RESTFRQ/CRVAL3)")
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
        self.map_for_testing = img_jypix  # for inspection
        uv_grid, du          = self.map_to_uvgrid(img_jypix, pix_deg)
        return {'uv': uv_grid, 'du': du, 'pixscale_deg': pix_deg}

    def sample_fft(self, model_name, data_name, field_key, spw_key):
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
        resid_vis = data_vis - model_vis

        # store sampled model
        sm_store = self.matched_models[model_name][data_name].setdefault('sampled_model', {})
        sm_store.setdefault(field_key, {})[spw_key] = {
            'model_vis': model_vis,
            'data_vis': data_vis,
            'resid_vis': resid_vis,
            'u': u,
            'v': v,
            'uvfreq': freq,
            'weights': wgt
        }
        return model_vis

    def map_to_vis(self, model_name, data_name, field_key, spw_key):
        """Convenience wrapper returning (model_vis, resid_vis)."""
        self.sample_fft(model_name, data_name, field_key, spw_key)
        entry = self.matched_models[model_name][data_name]['sampled_model'][field_key][spw_key]
        return entry['model_vis'], entry['resid_vis']