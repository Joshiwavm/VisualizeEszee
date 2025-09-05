import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from ..model.models import *
from ..model.unitwrapper import TransformInput  # renamed from transform
from ..utils import calculate_r500, ysznorm, cosmo

from typing import Tuple, Sequence, Dict, Any, Optional

## Marginalizer import removed (logic consolidated in MapMaking mixin)

class ModelHandler:
    """Handles model creation and sky map generation."""

    def add_model(self, name: str | None, source_type: str | None, model_type: str | None = None,
                    parameters: dict | None = None, quantiles: Sequence[float] | None = None, marginalized: bool = False, 
                    filename: str | None = None):
        """Register a model."""
        
        # Core model record
        if source_type == 'parameters':
            if model_type is None:
                raise ValueError("model type required to make a model")
            if parameters is None:
                raise ValueError("parameters required to make a model")
            self.models[name] = {
                'source': 'parameters',
                'type': model_type,
                'marginalized': False,
                'parameters': parameters,
                'calibration': [],  # default calibration
            }

        elif source_type == 'pickle':
            
            if filename is None:
                raise ValueError("filename required for pickle source_type")

            # Normalize data_name to list
            if isinstance(quantiles, (list, tuple)):
                quantiles = list(quantiles)
            else:
                quantiles = [quantiles]

            if marginalized and quantiles[0] is not None:
                raise ValueError("Cannot specify both marginalized and quantiles")
            
            elif not marginalized:
                parameters, calibs = self.get_parameters_from_quantiles(filename, quantiles)

                n_quants = len(parameters)
                n_compts = len(parameters[0]) if n_quants > 0 else 0
                for i_quant in range(n_quants):
                    for j_compt in range(n_compts):
                        self.models[f'{name}_q{quantiles[i_quant]}_c{j_compt}'] = {
                            'source': 'pickle',
                            'filename': filename,
                            'type': parameters[i_quant][j_compt]['model']['type'],
                            'quantile': quantiles[i_quant],
                            'component': j_compt,
                            'marginalized': False,
                            'parameters': parameters[i_quant][j_compt],
                            'calibration': calibs[i_quant],  # default calibration
                        }
            else:
                self.models[name] = {
                        'source': 'pickle',
                        'filename': filename,
                        'marginalized': True,
                        'calibration': [],  # default calibration

                    }

        else:
            raise ValueError("Invalid source_type, must be 'parameters' or 'pickle'")


    def add_model_maps(self, name: str, dataset_name: str, **kwargs):
        """Build model maps for model[name][dataset_name] (fields/spws level)."""
        
        model_info = self.models.get(name)
        dmeta = self.uvdata[dataset_name].get('metadata')
        fields = dmeta.get('fields')
        spws_nested = dmeta.get('spws')

        maps = {}
        for f, field in enumerate(fields):
            field_key = f'field{field}'
            maps.setdefault(field_key, {})
            for sdx, spw in enumerate(spws_nested[f]):
                spw_key = f'spw{spw}'
                binvis = dmeta.get('binvis')
                if binvis is not None:
                    fits_file = binvis.replace('fid', str(field)).replace('sid', str(spw)) + '.image.fits'
                else:
                    fits_file = f"output/{dmeta.get('array')}/output_{dmeta.get('band')}_{dmeta.get('array')}.im.field-{field}.spw-{spw}.image.fits"

                with fits.open(fits_file) as hdul:
                    header = hdul[0].header.copy()
                    image_data = hdul[0].data.copy()
                    nx, ny = header['NAXIS1'], header['NAXIS2']
                    cdelt1, cdelt2 = header['CDELT1'], header['CDELT2']
                    crval1, crval2 = header['CRVAL1'], header['CRVAL2']

                x_coords = np.arange(-nx/2, nx/2, 1.) + 0.5
                y_coords = np.arange(-ny/2, ny/2, 1.) - 0.5
                X, Y = np.meshgrid(x_coords, y_coords)
                ra_map = -1 * X * np.abs(cdelt1) / np.cos(np.deg2rad(crval2)) + crval1
                dec_map = Y * np.abs(cdelt2) + crval2


                if sdx == 0:
                    model_map = self.get_map(
                        model_info, ra_map, dec_map, header
                    )
                
                pbeam_file = fits_file.replace('.image.fits', '.pbeam.fits')

                with fits.open(pbeam_file) as pbeam_hdul:
                    pbeam_data = pbeam_hdul[0].data
                    model_map_corrected = model_map * pbeam_data

                maps[field_key][spw_key] = {
                    'model_data': model_map_corrected,
                    'image_data': image_data,
                    'pbeam_data': pbeam_data,
                    'header': header,
                }
                
        return maps

    # Deprecated private generation helpers removed (delegated to MapMaking)




    # TODO:
    def _generate_point_source(self, parameters, ra_map, dec_map):
        """Generate point source model."""
        ra_center = parameters.get('ra', parameters.get('RA', 0.0))
        dec_center = parameters.get('dec', parameters.get('Dec', 0.0))
        amplitude = parameters.get('amplitude', parameters.get('Amplitude', 1.0))
        
        # Find closest pixel to source position
        r_distance = np.sqrt((ra_map - ra_center)**2 + (dec_map - dec_center)**2)
        min_idx = np.unravel_index(np.argmin(r_distance), r_distance.shape)
        
        # Create delta function at source position
        model_map = np.zeros_like(ra_map)
        model_map[min_idx] = amplitude
        
        return model_map

    def _generate_gaussian_model(self, parameters, r_grid):
        """Generate Gaussian model on radial grid."""
        amplitude = parameters.get('amplitude', parameters.get('Amplitude', 1.0))
        major_axis = parameters.get('major', parameters.get('Major', 1.0))
        
        # Convert to radians if needed
        if major_axis > 0.01:
            major_axis_rad = np.deg2rad(major_axis / 60.0)  # Assume arcmin
        else:
            major_axis_rad = major_axis
            
        # Gaussian profile
        model_map = amplitude * np.exp(-0.5 * (r_grid / major_axis_rad)**2)
        
        return model_map