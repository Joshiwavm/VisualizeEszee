from __future__ import annotations

import os
import numpy as np
from astropy.io import fits

from VisualizeEszee.utils.utils import JyBeamToJyPix, smooth, extract_plane, get_map_beam_and_pix


class Deconvolve:
    """Deconvolution helper that implements the JvM-style clean."""

    def JvM_clean(self, model_name: str | None = None, data_name: str | None = None, *, notes=None, save_output=None):
        """Perform the same Jansky/Beam combination previously living on Manager.

        Parameters match the old Manager.JvM_clean and the method operates
        on `self` which is expected to provide the Manager-like attributes.
        """
        if model_name is None or data_name is None:
            raise ValueError("model_name and data_name must be provided for JvM_clean")
        if model_name not in self.matched_models:
            raise ValueError(f"Model '{model_name}' not found in matched_models")

        # Normalize data_name to list
        if isinstance(data_name, (list, tuple)):
            data_names = list(data_name)
        else:
            data_names = [data_name]

        # Find smallest beam across the supplied datasets
        target_bmaj = None
        target_bmin = None
        target_area = np.inf
        best_entry = None
        best_dn = None
        best_field = None
        best_spw = None

        for dn in data_names:
            assoc_dn = self.matched_models[model_name][dn]
            maps_dn = assoc_dn.get('maps', {})

            fk = next(iter(maps_dn))
            spw_dict = maps_dn[fk]
            sk = next(iter(spw_dict))
            entry_dn = spw_dict[sk]
            header_dn = entry_dn.get('header', {})

            bmaj_deg, bmin_deg, _, _ = get_map_beam_and_pix(header_dn)
            area = float(bmaj_deg) * float(bmin_deg)
            if area < target_area:
                target_area = area
                target_bmaj = bmaj_deg
                target_bmin = bmin_deg
                best_entry = entry_dn
                best_dn = dn
                best_field = fk
                best_spw = sk

        # Convert model map to Jy/pix using existing Fourier helper on self
        model_jypix, pix_deg = self._convert_model_map_to_jypix(model_name, best_entry)

        # Read pixel sizes from header
        header = best_entry.get('header', {})
        _,_,ipix_deg,jpix_deg = get_map_beam_and_pix(header)

        # Primary beam lookup (stored in matched map)
        maps_entry = self.matched_models[model_name][best_dn]['maps'][best_field][best_spw]
        pb_arr = maps_entry['pbeam_data']
        pb = extract_plane(pb_arr)

        pb_safe = np.where(pb == 0.0, 1.0, pb)

        # Divide out PB so smoothing acts on intrinsic sky signal (Jy/pix)
        model_jypix_nopb = model_jypix / pb_safe

        # Convert Jy/pixel -> Jy/beam using target beam
        factor = JyBeamToJyPix(ipix_deg, jpix_deg, float(target_bmaj), float(target_bmin))
        model_jybeam = model_jypix_nopb / factor

        # derive sigma (pixels) from geometric mean FWHM
        fwhm_target_deg = np.sqrt(float(target_bmaj) * float(target_bmin))
        fwhm_pix = fwhm_target_deg / ipix_deg
        sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Smooth in Jy/beam and re-apply PB to get final smoothed Jy/beam image
        model_smoothed_beam = smooth(model_jybeam, sigma)
        model_smoothed = model_smoothed_beam * pb

        # Build residual dirty image using sampled visibilities on the chosen grid
        npix = model_smoothed.shape[0]
        pixel_scale_deg = ipix_deg
        resid = self.multivis_to_image(model_name,
                                       data_name,
                                       use='resid',
                                       fields=['field0'],
                                       npix=npix,
                                       pixel_scale_deg=pixel_scale_deg,
                                       calib=1.0,
                                       align=True)

        jvm_image = resid + model_smoothed

        # Store result
        assoc = self.matched_models[model_name].setdefault(best_dn, {})
        assoc.setdefault('JvM_clean', {})
        assoc['deconvolved'] = jvm_image.astype(np.float32)

        if save_output is not None:
            os.makedirs(save_output, exist_ok=True)
            h = header.copy(); h['BUNIT'] = 'Jy/beam'
            fname = os.path.join(save_output, f"{model_name}_{best_dn}_{best_field}_{best_spw}_JvM_clean.fits")
            print(f"Writing JvM-cleaned image to {fname}")
            fits.writeto(fname, jvm_image.astype(np.float32), header=h, overwrite=True)

        return jvm_image
