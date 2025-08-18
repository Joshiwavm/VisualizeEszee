import os
import numpy as np
from collections import namedtuple
from astropy.io import fits

from ..utils.utils import *

# Define UVData namedtuple for storing uv-data
UVData = namedtuple('UVData', ['uwave', 'vwave', 'uvreal', 'uvimag', 'suvwght', 'uvfreq'])

class DataHandler:
    """Handles loading and storing of ALMA uv-data and ACT data via self.data dict."""
    def __init__(self):
        # Expect PlotManager to have initialized self.data; fallback if used standalone
        if not hasattr(self, 'data'):
            self.data = {'uv': {}, 'act': {}}

    def add_data(self, name, obstype, **kwargs):
        """
        Loads a uvdata set and stores it under the given name.
        Accepts all additional parameters as keyword arguments.
        fields: single field or list of fields
        spws: single spw, list of spws, or nested list [[spws_for_field0], [spws_for_field1], ...]
        """

        if obstype.lower() == 'interferometer':

            band = kwargs.get('band', None)
            array = kwargs.get('array', None)
            fields = kwargs.get('fields', None)
            spws = kwargs.get('spws', None)
            binvis = kwargs.get('binvis', None)

            # Normalize fields to list
            if not isinstance(fields, list):
                fields = [fields]
            
            # Handle spws structure - similar to add_model
            if not isinstance(spws, list):
                # Single spw for all fields
                spws_nested = [[spws] for _ in fields]
            elif len(spws) > 0 and not isinstance(spws[0], list):
                # Flat list of spws - apply to all fields
                spws_nested = [spws for _ in fields]
            else:
                # Already nested list
                spws_nested = spws
                
            if len(spws_nested) != len(fields):
                raise ValueError(f"Length of spws ({len(spws_nested)}) must match length of fields ({len(fields)})")

            print(f"Loading data for {len(fields)} field(s) with structure: {dict(zip(fields, spws_nested))}")

            uvdata = {}
            for f, field in enumerate(fields):
                print(f"  Processing field {field} with spws: {spws_nested[f]}")
                uvdata[f'field{field}'] = {}
                for spw in spws_nested[f]:
                    outvis = binvis.replace('-fid', f'-{field}').replace('-sid', f'-{spw}')
                    uvload = np.load(f'{outvis.replace(".ms.", ".im.")}.data.npz', fix_imports=True, encoding='bytes')
                    uvdata[f'field{field}'][f'spw{spw}'] = UVData(*[np.copy(uvload[uvload.files[0]][idx].flatten()) for idx in range(6)])
            
                # Get phase center from FITS header (using last spw for this field)
                last_spw = spws_nested[f][-1]
                fits_file = f'{outvis.replace("-sid", f"-{last_spw}").replace(".ms.", ".im.")}.image.fits'
                with fits.open(fits_file) as hdul:
                    header = hdul[0].header
                    ra = header.get('CRVAL1', None)
                    dec = header.get('CRVAL2', None)
                    phase_center = (ra, dec)
                uvdata[f'field{field}'][f'phase_center'] = phase_center

            self.data['uv'][name] = uvdata

            metadata = {
                'obstype': obstype,
                'band': band,
                'array': array,
                'fields': fields,
                'spws': spws_nested,  # Store the normalized nested structure
                'binvis': binvis,
            }
            self.data['uv'][name]['metadata'] = metadata
            print(f"Data loaded successfully for dataset '{name}' with {len(fields)} field(s)")
                

        if obstype.lower() == 'act':

            fdir = kwargs.get('fdir', None)

            ACTData = namedtuple('ACTData', ['filename', 'std'])
            dirs = os.listdir(fdir)
            coadd_ivar_files = [f for f in dirs if 'coadd_ivar' in f]
            actdata = {}
            for fname in coadd_ivar_files:
                freq_part = fname.split('_f')[1]
                freq = freq_part.split('_')[0]
                pa_part = fname.split('_pa')[1]
                arr = 'pa' + pa_part.split('_')[0]
                if freq not in actdata:
                    actdata[freq] = {}
                actdata[freq][arr] = ACTData(filename=fname, std=0.0)

            for freq in actdata:
                for arr in actdata[freq]:
                    file = os.path.join(fdir, actdata[freq][arr].filename)
                    data_image, header = fits.getdata(file, header=True)
                    if freq == '090':
                        norm = KcmbToJyPix(90e9, header['CDELT1'], header['CDELT2'])  *1e-6
                        npix = np.pi * 2.* 2./(4*np.log(2)) / (np.abs(header['CDELT1']*header['CDELT2'])*3600)
                    elif freq == '150':
                        norm = KcmbToJyPix(150e9, header['CDELT1'], header['CDELT2']) *1e-6
                        npix = np.pi * 1.4* 1.4/(4*np.log(2)) / (np.abs(header['CDELT1']*header['CDELT2'])*3600)
                    else:
                        norm = 1.0
                    std = np.mean(data_image) ** -0.5 * norm / npix
                    actdata[freq][arr] = actdata[freq][arr]._replace(std=std)
            self.data['act'][name] = actdata

            if 'metadata' not in self.data['act'][name]:
                self.data['act'][name]['metadata'] = {
                    'obstype': obstype,
                    'fdir': fdir,
            }
