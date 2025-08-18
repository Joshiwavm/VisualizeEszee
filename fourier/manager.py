from __future__ import annotations
import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple, Sequence, Dict, Any, Optional

class FourierManager:
    """Fourier-domain operations using matched_models structure."""

    @staticmethod
    def map_to_uvgrid(image: np.ndarray, pixel_scale_deg: float) -> Tuple[np.ndarray, float]:
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError("image must be square 2D array")
        N = image.shape[0]
        delt_rad = np.deg2rad(abs(pixel_scale_deg))
        du = 1.0 / (N * delt_rad)
        uv_rfft_shifted = np.fft.fftshift(np.fft.rfft2(np.fft.fftshift(image)), axes=0)
        return uv_rfft_shifted, du

    @staticmethod
    def sample_uv(uv_rfft_shifted: np.ndarray, u: Sequence[float], v: Sequence[float], du: float,
                  dRA: float = 0.0, dDec: float = 0.0, PA: float = 0.0, origin: str = 'upper') -> np.ndarray:
        u = np.asarray(u); v = np.asarray(v)
        if u.shape != v.shape:
            raise ValueError("u and v must have same shape")
        if origin not in ('upper','lower'):
            raise ValueError("origin must be 'upper' or 'lower'")
        v_origin = 1.0 if origin == 'upper' else -1.0
        nxy = uv_rfft_shifted.shape[0]
        if uv_rfft_shifted.shape[1] != nxy//2 + 1:
            raise ValueError("uv_rfft_shifted second dimension inconsistent with first")
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
        if not use_jax:
            raise ValueError("vis_to_image requires JAX backend.")
        import jax  # noqa: F401
        import jax_finufft
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
        data_vis = np.asarray(data_vis); model_vis = np.asarray(model_vis)
        if data_vis.shape != model_vis.shape:
            raise ValueError("data_vis and model_vis shape mismatch")
        return data_vis - model_vis

    @staticmethod
    def scale_vis(vis: Sequence[complex], factor: float) -> np.ndarray:
        return np.asarray(vis) * factor

    @staticmethod
    def phase_shift(vis: Sequence[complex], u: Sequence[float], v: Sequence[float],
                    dRA: float = 0.0, dDec: float = 0.0) -> np.ndarray:
        vis = np.asarray(vis); u = np.asarray(u); v = np.asarray(v)
        if vis.shape != u.shape:
            raise ValueError("vis and u/v shapes mismatch")
        phase = np.exp(2.0 * np.pi * 1j * (u * dRA + v * dDec))
        return vis * phase

    @staticmethod
    def _extract_plane(arr):
        a = np.asarray(arr)
        if a.ndim == 4:
            return a[0, 0]
        if a.ndim == 3:
            return a[0]
        return a

    def _convert_model_map_to_jybeam(self, entry):
        from ..utils.utils import ytszToJyPix, JyBeamToJyPix
        header = entry['header']
        model_plane = self._extract_plane(entry['model_data'])
        model_plane = np.nan_to_num(model_plane, nan=0.0, posinf=0.0, neginf=0.0)
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
        jy_pix = model_plane * ytszToJyPix(freq, ipix_deg, jpix_deg)
        bmaj = header.get('BMAJ'); bmin = header.get('BMIN')
        if not bmaj or not bmin:
            raise ValueError('Beam parameters (BMAJ/BMIN) required for Jy/beam conversion.')
        beam_conv = JyBeamToJyPix(ipix_deg, jpix_deg, bmaj, bmin)
        return jy_pix / beam_conv, ipix_deg

    # ------------------------------------------------------------------
    # New sampling API against matched_models structure
    # ------------------------------------------------------------------
    def fft_map(self, model_name, data_name, field_key, spw_key):
        maps = self.matched_models[model_name][data_name]['maps']
        entry = maps[field_key][spw_key]
        jy_beam_image, pix_deg = self._convert_model_map_to_jybeam(entry)
        uv_grid, du = self.map_to_uvgrid(jy_beam_image, pix_deg)
        return {'uv': uv_grid, 'du': du, 'pixscale_deg': pix_deg}

    def sample_fft(self, model_name, data_name, field_key, spw_key):
        uvrec = self.uvdata[data_name][field_key][spw_key]
        u, v = uvrec.uwave, uvrec.vwave
        real, imag, wgt = uvrec.uvreal, uvrec.uvimag, uvrec.suvwght
        uv_entry = self.fft_map(model_name, data_name, field_key, spw_key)
        model_vis = self.sample_uv(uv_entry['uv'], u, v, uv_entry['du'])
        data_vis = real + 1j * imag
        resid_vis = data_vis - model_vis
        sm_store = self.matched_models[model_name][data_name].setdefault('sampled_model', {})
        sm_store.setdefault(field_key, {})[spw_key] = {
            'model_vis': model_vis,
            'data_vis': data_vis,
            'resid_vis': resid_vis,
            'u': u,
            'v': v,
            'weights': wgt
        }
        return model_vis

    def map_to_vis(self, model_name, data_name, field_key, spw_key):
        self.sample_fft(model_name, data_name, field_key, spw_key)
        entry = self.matched_models[model_name][data_name]['sampled_model'][field_key][spw_key]
        return entry['model_vis'], entry['resid_vis']