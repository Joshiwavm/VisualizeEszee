from __future__ import annotations

import os
import numpy as np
import warnings
from astropy.io import fits
from reproject import reproject_interp

from VisualizeEszee.utils.utils import JyBeamToJyPix, smooth, extract_plane, get_map_beam_and_pix


class Deconvolve:
    """Deconvolution helper that implements the JvM-style clean."""

    def JvM_clean(self, model_name: str | None = None, data_name: str | None = None, taper:float|None=None, *, notes=None, save_output=None):
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
        # Smooth in Jy/beam and re-apply PB to get final smoothed Jy/beam image
        if taper is None:
            fwhm_target_deg = np.sqrt(float(target_bmaj) * float(target_bmin))
            fwhm_pix = fwhm_target_deg / ipix_deg
            sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        else:
            fwhm_pix = taper/3600.0/ipix_deg
            sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
 
        model_smoothed_beam = smooth(model_jybeam, sigma)

        # Multiply by the combined primary-beam map (use pb_combined for full multi-field response)
        pb_combined = self._get_field_averaged_pb(model_name, data_names, best_dn, best_field, best_spw)
        model_smoothed = model_smoothed_beam * pb_combined

        # Build residual dirty image using sampled visibilities on the chosen grid
        npix = model_smoothed.shape[0]
        pixel_scale_deg = ipix_deg
        resid = self.multivis_to_image(model_name,
                                       data_names,
                                       use='resid',
                                       npix=npix,
                                       pixel_scale_deg=pixel_scale_deg,
                                       calib=1.0,
                                       align=True, 
                                       taper=taper)

        jvm_image = resid + model_smoothed
        std = np.nanstd(resid)

        # Store result under a concatenated data-name key so multi-dataset runs
        # are discoverable by consumers (e.g. plot_map with data_name=[...])
        concat_dn = "+".join(data_names)
        assoc = self.matched_models[model_name].setdefault(concat_dn, {})
        assoc['deconvolved'] = jvm_image.astype(np.float32)
        assoc['std'] = std

        if save_output is not None:
            os.makedirs(save_output, exist_ok=True)
            h = header.copy(); h['BUNIT'] = 'Jy/beam'; h['HISTORY']='JvM-cleaned image'; h['rms']=(std, 'Jy/beam')
            fname = os.path.join(save_output, f"{model_name}_{concat_dn}_{best_field}_{best_spw}_JvM_clean.fits")
            print(f"Writing JvM-cleaned image to {fname}")
            fits.writeto(fname, jvm_image.astype(np.float32), header=h, overwrite=True)

        return jvm_image

    def _likelihood_call(self):
        """--- IGNORE ---"""
        pass

    def _get_field_averaged_pb(self, model_name: str, data_names: list, best_dn: str, best_field: str, best_spw: str):
        """Build a combined primary-beam map for the provided model/data selection. """
        # Locate the reference header / shape from best entry
        ref_entry = self.matched_models[model_name][best_dn]['maps'][best_field][best_spw]
        ref_header = ref_entry.get('header', None)
        ref_plane = extract_plane(ref_entry.get('pbeam_data'))
        ny, nx = np.asarray(ref_plane).shape

        # Accumulators
        acc_pb2 = np.zeros((ny, nx), dtype=np.float64)

        # Iterate over datasets & fields
        for dn in data_names:
            maps_dn = self.matched_models[model_name].get(dn, {}).get('maps', {})
            for fk, spw_dict in maps_dn.items():
                # Use first spw key instead of best_spw
                first_spw = next(iter(spw_dict))
                entry = spw_dict[first_spw]
                pb_arr = entry.get('pbeam_data')
                src_header = entry.get('header', None)
                
                # Suppress FITS WCS warnings during reprojection
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*FITSFixedWarning.*')
                    warnings.filterwarnings('ignore', module='astropy.wcs')
                    reprojected, _ = reproject_interp((pb_arr, src_header), ref_header)
                reprojected = np.nan_to_num(reprojected, nan=0.0, posinf=0.0, neginf=0.0)

                # accumulate PB squared
                pb_plane = extract_plane(reprojected)
                acc_pb2 += pb_plane ** 2.0

        # rescale to max of 1
        pb_combined = acc_pb2/np.nanmax(acc_pb2)
        return pb_combined