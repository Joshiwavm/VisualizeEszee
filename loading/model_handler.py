import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from ..model.models import *

class ModelHandler:
    """
    Handles model creation and sky map generation.
    """
    def __init__(self):
        """Initialize model storage containers."""
        self.models = {}
        self.model_maps = {}

    def add_model(self, name, source_type, band, array, fields, spws, binvis=None, **kwargs):
        """
        Add a model and create sky maps using FITS header structure.
        source_type: 'parameters' or 'pickle'
        band: band name (e.g., 'band1', 'band3')
        array: array name (e.g., 'com07m', 'com12m')
        fields: single field or list of fieldss
        spws: single spw, list of spws, or nested list [[spws_for_field0], [spws_for_field1], ...]
        binvis: base path for visibility/FITS files (e.g., '../output/com12m/output_band1_com12m.ms.field-fid.spw-sid')
        """
        
        # Normalize fields and spws to lists
        if not isinstance(fields, list):
            fields = [fields]
        
        # Handle spws structure - similar to uvdata
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
        
        # Store model info first (before creating maps)
        if source_type == 'parameters':
            model_type = kwargs.get('model_type', 'custom')
            parameters = kwargs.get('parameters', {})
            
            self.models[name] = {
                'source': 'parameters',
                'type': model_type,
                'parameters': parameters,
                'band': band,
                'array': array,
                'fields': fields,
                'spws': spws_nested,
                'binvis': binvis
            }
            
        elif source_type == 'pickle':
            filename = kwargs.get('filename', None)
            major_indices = kwargs.get('major_indices', [None])
            flux_indices = kwargs.get('flux_indices', [None])
            
            if filename is None:
                raise ValueError("filename required for pickle source_type")
                
            from ..model.models import get_samples
            edges = get_samples(filename, major=major_indices, flux=flux_indices)
            
            self.models[name] = {
                'source': 'pickle',
                'filename': filename,
                'quantiles': edges,
                'major_indices': major_indices,
                'flux_indices': flux_indices,
                'band': band,
                'array': array,
                'fields': fields,
                'spws': spws_nested,
                'binvis': binvis
            }
            
        else:
            raise ValueError("source_type must be 'parameters' or 'pickle'")

        self.add_model_maps(name, **kwargs)
        
    def add_model_maps(self, name: str, **kwargs):

        # Create nested dictionary structure for model maps
        self.model_maps[name] = {}
        
        # Loop over fields and spws to create model maps
        for f, field in enumerate(self.models[name]['fields']):
            field_key = f'field{field}'
            self.model_maps[name][field_key] = {}
            
            for spw in self.models[name]['spws'][f]:
                spw_key = f'spw{spw}'
                                
                # Load FITS header for this field/spw combination
                if self.models[name]['binvis'] is not None:
                    # Use provided binvis path, replace placeholders with actual field/spw
                    fits_file = self.models[name]['binvis'].replace('fid', str(field)).replace('sid', str(spw)) + '.image.fits'
                else:
                    # Fallback to default path construction
                    fits_file = f'output/{self.models[name]['array']}/output_{self.models[name]['band']}_{self.models[name]['array']}.im.field-{field}.spw-{spw}.image.fits'
                
                try:
                    with fits.open(fits_file) as hdul:
                        header = hdul[0].header.copy()
                        image_data = hdul[0].data.copy()  # Save the image data
                        nx, ny = header['NAXIS1'], header['NAXIS2']
                        crpix1, crpix2 = header['CRPIX1'], header['CRPIX2']
                        crval1, crval2 = header['CRVAL1'], header['CRVAL2']
                        cdelt1, cdelt2 = header['CDELT1'], header['CDELT2']
                                            
                except FileNotFoundError:
                    print(f"  Warning: FITS file not found: {fits_file}")
                    continue
                
                # Create coordinate grids for this field/spw following Veszee approach
                x_coords = np.arange(-nx/2, nx/2, 1.)
                y_coords = np.arange(-ny/2, ny/2, 1.)
                x_coords += 0.5
                y_coords -= 0.5
                X, Y = np.meshgrid(x_coords, y_coords)
                
                # Convert pixel coordinates to sky coordinates using WCS transformation
                ra_map = -1 * X * np.abs(cdelt1) / np.cos(np.deg2rad(crval2)) + crval1
                dec_map = Y * np.abs(cdelt2) + crval2
                
                # Generate model map based on source type
                if self.models[name]['source'] == 'parameters':
                    model_map = self._generate_model_from_parameters(
                        self.models[name]['type'], 
                        self.models[name]['parameters'], 
                        ra_map, dec_map, header
                    )
                    
                elif self.models[name]['source'] == 'pickle':
                    quantile_type = kwargs.get('quantile_type', '50th')
                    n_samples = kwargs.get('n_samples', 100)
                    
                    if quantile_type == 'marginalized':
                        model_map = self._generate_marginalized_model(
                            self.models[name], ra_map, dec_map, header, n_samples
                        )
                    else:
                        model_map = self._generate_model_from_quantiles(
                            self.models[name], quantile_type, ra_map, dec_map, header
                        )
                
                # Load and apply primary beam correction
                pbeam_file = fits_file.replace('.image.fits', '.pbeam.fits')
                try:
                    with fits.open(pbeam_file) as pbeam_hdul:
                        pbeam_data = pbeam_hdul[0].data
                        # print(f"  Applying primary beam correction from: {pbeam_file}")
                        
                        # Apply primary beam correction to model map
                        model_map = model_map * pbeam_data
                        
                except FileNotFoundError:
                    print(f"  Warning: Primary beam file not found: {pbeam_file}")
                    print(f"  Continuing without primary beam correction")
                except Exception as e:
                    print(f"  Warning: Error loading primary beam: {e}")
                    print(f"  Continuing without primary beam correction")

                # Store the model map in nested structure
                self.model_maps[name][field_key][spw_key] = {
                    'model_data': model_map,
                    'image_data': image_data,  # Store the original image data for comparison
                    'header': header,
                    'ra_map': ra_map,
                    'dec_map': dec_map,
                    'pb_map': pbeam_data
                }

    def _generate_model_from_parameters(self, model_type, parameters, ra_map, dec_map, header):
        """Generate model map from direct parameters using proper radial grid approach."""
        
        # Create radial distance grid 
        r_grid = self._make_radial_grid(ra_map, dec_map, parameters['model'])

        # Generate model based on type
        if model_type in ['gnfwPressure', 'A10Pressure', 'betaPressure']:
            model_map = self._generate_pressure_profile(model_type, parameters['model'], r_grid, header)
        elif model_type == 'pointSource':
            model_map = self._generate_point_source(parameters['model'], ra_map, dec_map)
        elif model_type in ['gaussSource', 'gaussSurface']:
            model_map = self._generate_gaussian_model(parameters['model'], r_grid)
        else:
            print(f"Warning: Model type '{model_type}' not implemented, returning zeros")
            model_map = np.zeros_like(ra_map)
        
        return model_map

    def _make_radial_grid(self, ra_map, dec_map, model_params):
        """
        Create radial distance grid from RA/Dec maps using model center and orientation.
        This follows the Veszee approach for proper coordinate transformation.
        """
        
        # Extract model center and orientation parameters
        ra_center = model_params.get('ra', model_params.get('RA', 0.0))
        dec_center = model_params.get('dec', model_params.get('Dec', 0.0))
        angle = model_params.get('angle', model_params.get('Angle', 0.0)) 
        eccentricity = model_params.get('e', model_params.get('eccentricity', 0.0))
        
        # Pre-compute trigonometric functions
        cosy = np.cos(np.deg2rad(dec_center))
        cost = np.cos(np.deg2rad(angle))
        sint = np.sin(np.deg2rad(angle))
        
        # Transform to model-centered coordinate system
        modgrid_x = (-(ra_map - ra_center) * cosy * sint - (dec_map - dec_center) * cost)
        modgrid_y = ((ra_map - ra_center) * cosy * cost - (dec_map - dec_center) * sint)
        
        # Calculate elliptical radial distance
        r = np.sqrt(modgrid_x**2 + modgrid_y**2 / (1.0 - eccentricity)**2)
        
        return r

    def _generate_pressure_profile(self, model_type, parameters, r_grid, header):
        """Generate pressure profile model using Veszee approach with proper model functions."""

        # Extract basic parameters
        amplitude = parameters.get('amplitude', parameters.get('Amplitude', 1.0))
        major_axis_deg = parameters.get('major', parameters.get('Major', 1.0))
        redshift = parameters.get('redshift', parameters.get('z', 0.5))
        
        # Convert major axis to degrees if needed (assume arcmin if > 1)
        if major_axis_deg > 1.0:
            major_axis_deg = major_axis_deg / 60.0
            
        # Apply the Veszee coordinate transformation to get physical coordinates
        r_physical = np.deg2rad(r_grid) * cosmo.angular_diameter_distance(redshift)
        major_axis_physical = np.deg2rad(major_axis_deg) * cosmo.angular_diameter_distance(redshift)
        coord = r_physical.value / major_axis_physical.value
                
        # Create radial mesh for profile evaluation (like Veszee info.linmesh)
        r_min = 1e-3  # Minimum radius to avoid singularities
        r_max = coord.max() * 2.0  # Extend beyond image
        n_points = 200  # Number of radial points
        
        rs = np.logspace(np.log10(r_min), np.log10(r_max), n_points)  # Start from small radius
                    
        # Prepare parameters for model functions (following Veszee input_par format)
        if model_type == 'gnfwPressure':
            # gnfwProfile(grid, offset, amp, major, e, alpha, beta, gamma, limdist, epsrel, freeLS)
            offset = parameters.get('offset')
            e = parameters.get('e', parameters.get('eccentricity'))
            alpha = parameters.get('alpha')
            beta = parameters.get('beta')
            gamma = parameters.get('gamma')
            
            # Evaluate profile on radial mesh
            integrated_P = gnfwProfile(rs, offset, amplitude, 1.0, e, alpha, beta, gamma)
            
        elif model_type == 'A10Pressure':
            # a10Profile(grid, offset, amp, major, e, alpha, beta, gamma, ap, c500, mass, limdist, epsrel, freeLS)
            offset = parameters.get('offset', 0.0)
            e = parameters.get('e', parameters.get('eccentricity', 0.0))
            alpha = parameters.get('alpha', 1.05)
            beta = parameters.get('beta', 5.49)
            gamma = parameters.get('gamma', 0.31)
            concentration = parameters.get('concentration', parameters.get('c500', 1.18))
            alpha_p = parameters.get('alpha_p', 0.12)
            mass = parameters.get('mass', 2e14)
            
            # Convert mass to proper units (like in Veszee)
            mass_norm = mass / 3e14  # Normalize by 3e14 solar masses
            
            # Evaluate profile on radial mesh
            integrated_P = a10Profile(rs, offset, amplitude, 1.0, e, alpha, beta, gamma, 
                                    alpha_p, concentration, mass_norm)
            
        elif model_type == 'betaPressure':
            # betaProfile(grid, offset, amp, major, e, beta, limdist, epsrel, freeLS)
            offset = parameters.get('offset', 0.0)
            e = parameters.get('e', parameters.get('eccentricity', 0.0))
            beta = parameters.get('beta', 0.7)
            
            # Evaluate profile on radial mesh  
            integrated_P = betaProfile(rs, offset, amplitude, 1.0, e, beta)
            
        else:
            print(f"  Warning: Model type '{model_type}' not recognized")
            return np.zeros_like(r_grid)
                
        # Interpolate profile onto coordinate grid (like in Veszee)
        model_map = np.interp(coord, rs, integrated_P)
        
        # Apply SZ normalization if available (like in Veszee: * self.info.ysznorm.value)
        if 'ysznorm' in parameters:
            model_map *= parameters['ysznorm']
        elif hasattr(parameters, 'ysznorm'):
            model_map *= parameters.ysznorm.value
                    
        return model_map

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