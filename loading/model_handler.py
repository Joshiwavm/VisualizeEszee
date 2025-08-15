import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const
from ..model.models import *
from ..model.unitwrapper import TransformInput  # renamed from transform
from ..utils import calculate_r500, ysznorm

class ModelHandler:
    """Handles model creation and sky map generation."""
    def __init__(self):
        self.models = {}
        self.model_maps = {}

    def add_model(self, name, source_type, model_type=None, parameters=None,
                  band=None, array=None, fields=None, spws=None, binvis=None, data_name: str = None, **kwargs):
        """Register a model.

        If immediate spatial metadata provided (band/array/fields/spws), they are stored
        under key (data_name if given else '__legacy__'). Additional datasets added later
        via match_model() appear as new keys at models[name][data_name].
        """
        if source_type not in ('parameters', 'pickle'):
            raise ValueError("source_type must be 'parameters' or 'pickle'")
        immediate_maps = all(v is not None for v in (band, array, fields, spws))
        spws_nested = None
        if immediate_maps:
            if not isinstance(fields, list):
                fields = [fields]
            if not isinstance(spws, list):
                spws_nested = [[spws] for _ in fields]
            elif len(spws) > 0 and not isinstance(spws[0], list):
                spws_nested = [spws for _ in fields]
            else:
                spws_nested = spws
            if len(spws_nested) != len(fields):
                raise ValueError(f"Length of spws ({len(spws_nested)}) must match length of fields ({len(fields)})")
        # Core model record
        if source_type == 'parameters':
            if model_type is None:
                model_type = kwargs.get('model_type', 'custom')
            if parameters is None:
                parameters = kwargs.get('parameters', {})
            self.models[name] = {
                'source': 'parameters',
                'type': model_type,
                'parameters': parameters,
            }
        else:
            filename = kwargs.get('filename')
            if filename is None:
                raise ValueError("filename required for pickle source_type")
            major_indices = kwargs.get('major_indices', [None])
            flux_indices = kwargs.get('flux_indices', [None])
            from ..model.models import get_samples
            edges = get_samples(filename, major=major_indices, flux=flux_indices)
            self.models[name] = {
                'source': 'pickle',
                'filename': filename,
                'quantiles': edges,
                'major_indices': major_indices,
                'flux_indices': flux_indices,
            }
        # Attach immediate spatial metadata under chosen key
        if immediate_maps:
            key = data_name if data_name else '__legacy__'
            self.models[name][key] = {
                'band': band,
                'array': array,
                'fields': fields,
                'spws': spws_nested,
                'binvis': binvis,
            }
            self.add_model_maps(name, dataset_name=key, **kwargs)
        else:
            self.model_maps.setdefault(name, {})
        return self.models[name]

    def add_model_maps(self, name: str, dataset_name: str, **kwargs):
        """Build model maps for model[name][dataset_name] (fields/spws level)."""
        model_info = self.models.get(name)
        if not model_info:
            raise ValueError(f"Model '{name}' not registered.")
        dmeta = model_info.get(dataset_name)
        if dmeta is None:
            raise ValueError(f"Dataset '{dataset_name}' metadata missing for model '{name}'.")
        fields = dmeta.get('fields')
        spws_nested = dmeta.get('spws')
        if fields is None or spws_nested is None:
            raise ValueError(f"Model '{name}' dataset '{dataset_name}' missing fields/spws; cannot build maps.")
        self.model_maps.setdefault(name, {})
        self.model_maps[name].setdefault(dataset_name, {})
        for f, field in enumerate(fields):
            field_key = f'field{field}'
            self.model_maps[name][dataset_name].setdefault(field_key, {})
            for spw in spws_nested[f]:
                spw_key = f'spw{spw}'
                binvis = dmeta.get('binvis')
                if binvis is not None:
                    fits_file = binvis.replace('fid', str(field)).replace('sid', str(spw)) + '.image.fits'
                else:
                    fits_file = f"output/{dmeta.get('array')}/output_{dmeta.get('band')}_{dmeta.get('array')}.im.field-{field}.spw-{spw}.image.fits"
                try:
                    from astropy.io import fits
                    with fits.open(fits_file) as hdul:
                        header = hdul[0].header.copy()
                        image_data = hdul[0].data.copy()
                        nx, ny = header['NAXIS1'], header['NAXIS2']
                        cdelt1, cdelt2 = header['CDELT1'], header['CDELT2']
                        crval1, crval2 = header['CRVAL1'], header['CRVAL2']
                except FileNotFoundError:
                    print(f"  Warning: FITS file not found: {fits_file}")
                    continue
                import numpy as np
                x_coords = np.arange(-nx/2, nx/2, 1.) + 0.5
                y_coords = np.arange(-ny/2, ny/2, 1.) - 0.5
                X, Y = np.meshgrid(x_coords, y_coords)
                ra_map = -1 * X * np.abs(cdelt1) / np.cos(np.deg2rad(crval2)) + crval1
                dec_map = Y * np.abs(cdelt2) + crval2
                if model_info['source'] == 'parameters':
                    model_map = self._generate_model_from_parameters(
                        model_info['type'], model_info['parameters'], ra_map, dec_map, header
                    )
                else:
                    quantile_type = kwargs.get('quantile_type', '50th')
                    n_samples = kwargs.get('n_samples', 100)
                    if quantile_type == 'marginalized':
                        model_map = self._generate_marginalized_model(
                            model_info, ra_map, dec_map, header, n_samples
                        )
                    else:
                        model_map = self._generate_model_from_quantiles(
                            model_info, quantile_type, ra_map, dec_map, header
                        )
                pbeam_file = fits_file.replace('.image.fits', '.pbeam.fits')
                try:
                    with fits.open(pbeam_file) as pbeam_hdul:
                        pbeam_data = pbeam_hdul[0].data
                        model_map = model_map * pbeam_data
                except FileNotFoundError:
                    print(f"  Warning: Primary beam file not found: {pbeam_file}")
                    pbeam_data = np.ones_like(model_map)
                except Exception as e:
                    print(f"  Warning: Error loading primary beam: {e}")
                    pbeam_data = np.ones_like(model_map)
                self.model_maps[name][dataset_name][field_key][spw_key] = {
                    'model_data': model_map,
                    'image_data': image_data,
                    'header': header,
                    'ra_map': ra_map,
                    'dec_map': dec_map,
                    'pb_map': pbeam_data
                }



    def _generate_model_from_parameters(self, model_type, parameters, ra_map, dec_map, header,
                                        rs = np.append(0.0, np.logspace(-5, 5, 100))):
        """Generate model map from direct parameters."""

        xform = TransformInput(parameters['model'], model_type)
        input_par = xform.generate()
        
        r_grid = self._make_radial_grid(ra_map, dec_map, parameters['model'])

        z = parameters['model'].get('redshift', parameters['model'].get('z'))
        r_phys_mpc = np.deg2rad(r_grid) * cosmo.angular_diameter_distance(z).to(u.Mpc).value
        coord = r_phys_mpc / input_par.get('major')

        rs_sample = rs[1:] if model_type == 'gnfwPressure' else rs

        if model_type == 'A10Pressure':
            profile = a10Profile(rs_sample, 
                                 input_par.get('offset'), 
                                 input_par['amp'], 
                                 input_par.get('major'), 
                                 input_par.get('e'),
                                 input_par['alpha'], 
                                 input_par['beta'], 
                                 input_par['gamma'],
                                 input_par['ap'], 
                                 input_par['c500'], 
                                 input_par['mass'])
        elif model_type == 'gnfwPressure':
            profile = gnfwProfile(rs_sample, 
                                input_par.get('offset'), 
                                input_par['amp'], 
                                input_par.get('major'), 
                                input_par.get('e'),
                                input_par['alpha'], 
                                input_par['beta'], 
                                input_par['gamma'])
        elif model_type == 'betaPressure':
            profile = betaProfile(rs_sample, 
                                input_par.get('offset'), 
                                input_par['amp'], 
                                input_par.get('major'), 
                                input_par.get('e'), 
                                input_par['beta'])
        else:
            return np.zeros_like(r_grid)

        model_map = np.interp(coord, rs_sample, profile, left=profile[0], right=profile[-1])
        model_map = model_map * ysznorm

        return model_map

    def _make_radial_grid(self, ra_map, dec_map, model_params):
        """
        Create radial distance grid from RA/Dec maps using model center and orientation.
        This follows the Veszee approach for proper coordinate transformation.
        """
        
        # Extract model center and orientation parameters
        ra_center = model_params.get('ra')
        dec_center = model_params.get('dec')
        angle = model_params.get('angle', 0) 
        eccentricity = model_params.get('e', 0)
        
        # Pre-compute trigonometric functions
        cosy = np.cos(np.deg2rad(dec_center))
        cost = np.cos(np.deg2rad(angle))
        sint = np.sin(np.deg2rad(angle))
        
        # Transform to model-centered coordinate system
        modgrid_x = (-(ra_map - ra_center) * cosy * sint - (dec_map - dec_center) * cost)
        modgrid_y = ( (ra_map - ra_center) * cosy * cost - (dec_map - dec_center) * sint)
        
        # Calculate elliptical radial distance
        r = np.sqrt(modgrid_x**2 + modgrid_y**2 / (1.0 - eccentricity)**2)
        
        return r

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

    def _generate_model_from_quantiles(self, model, quantile_type, ra_map, dec_map, header):
        """Generate model map from specific quantile (placeholder)."""
        quantiles = model['quantiles']
        
        # Map quantile_type to index
        quantile_map = {'16th': 0, '50th': 1, '84th': 2}
        if quantile_type not in quantile_map:
            raise ValueError(f"quantile_type must be one of {list(quantile_map.keys())}")
        
        idx = quantile_map[quantile_type]
        
        # Extract parameters at specific quantile
        params_at_quantile = quantiles[:, idx]
                
        # Placeholder: create dummy model
        model_map = np.zeros_like(ra_map)
        
        # TODO: Implement model generation from quantile parameters
        # Need to map parameter indices to physical meaning based on model type
        
        return model_map

    def _generate_marginalized_model(self, model, ra_map, dec_map, header, n_samples):
        """Generate marginalized model over posterior samples (placeholder)."""
        from ..model.models import get_samples
        
        # Reload full samples for marginalization
        filename = model['filename']
        major_indices = model['major_indices']
        flux_indices = model['flux_indices']
                
        # TODO: Implement proper sample drawing and marginalization
        # This would involve:
        # 1. Loading full posterior samples
        # 2. Drawing n_samples from the posterior
        # 3. Generating model for each sample
        # 4. Computing mean/median of all models
        
        # Placeholder: create dummy model
        model_map = np.zeros_like(ra_map)
        
        return model_map