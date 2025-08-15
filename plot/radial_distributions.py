import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from ..utils.style import setup_plot_style

class PlotRadialDistributions:
    
    def find_central_field(self, band_name):
        """
        Find the field closest to the center of all fields.
        
        Parameters
        ----------
        band_name : str
            Name of the band to analyze
            
        Returns
        -------
        central_field : str
            Name of the central field
        """
        fields = list(self.uvdata[band_name].keys())
        fields = [f for f in fields if f != 'metadata']
        
        if len(fields) == 1:
            return fields[0]
            
        # Get phase centers for all fields
        ras = []
        decs = []
        
        for field in fields:
            phase_center = self.uvdata[band_name][field]['phase_center']
            ras.append(phase_center[0])  # RA
            decs.append(phase_center[1])  # Dec
            
        ras = np.array(ras)
        decs = np.array(decs)
        
        # Find center coordinates
        center_ra = np.mean(ras)
        center_dec = np.mean(decs)
        
        # Find field closest to center
        distances = np.sqrt((ras - center_ra)**2 + (decs - center_dec)**2)
        central_field_idx = np.argmin(distances)
        
        return fields[central_field_idx]

    def apply_phase_shift(self, dRA, dDec, field_data):
        """
        Apply a phase shift to the visibilities based on an offset in RA and Dec.

        Parameters
        ----------
        dRA : float
            Offset in Right Ascension (arcsec).
        dDec : float
            Offset in Declination (arcsec).
        field_data : dict
            Field data containing UV coordinates and visibilities

        Returns
        -------
        shifted_data : dict
            Copy of field data with phase-shifted visibilities
        """
        shifted_data = copy.deepcopy(field_data)

        # Convert the offsets from arcsec to radians
        dRA_rad = np.deg2rad(dRA / 3600.)
        dDec_rad = np.deg2rad(dDec / 3600.)

        # Phase shift each SPW
        for spw_name, spw_data in shifted_data.items():
            if spw_name == 'phase_center':
                continue
                
            # Form complex visibilities from original real and imaginary parts
            vis = spw_data.uvreal + 1j * spw_data.uvimag
            
            # Compute the phase factor (using (u, v) in wavelengths)
            phase_factor = np.exp(-2j * np.pi * (spw_data.uwave * dRA_rad +
                                                    spw_data.vwave * dDec_rad))
            
            # Apply the phase shift
            new_vis = vis * phase_factor

            # I need to a primairy beam correction!!!!!!
            
            # Store the phase-shifted visibilities
            shifted_data[spw_name] = spw_data._replace(
                uvreal=new_vis.real,
                uvimag=new_vis.imag
            )

        return shifted_data

    def get_binned_uvdatapoints(self, band_name, nbins=20, custom_phase_center=None):
        """
        Bin UV data points radially after phase shifting all fields to central field.
        
        Parameters
        ----------
        band_name : str
            Name of the band to analyze
        nbins : int, optional
            Number of bins for radial averaging (default: 20)
        custom_phase_center : tuple, optional
            Custom phase center (RA, Dec) in degrees to shift all fields to.
            If None, uses automatically determined central field.
            
        Returns
        -------
        tuple
            Binned real, real errors, imaginary, imaginary errors, bin edges, bin centers
        """
        # Determine phase center to use
        if custom_phase_center is not None:
            central_phase_center = custom_phase_center
            print(f"Using custom phase center: RA={central_phase_center[0]:.6f}, Dec={central_phase_center[1]:.6f}")
        else:
            # Find central field
            central_field = self.find_central_field(band_name)
            print(f"Using central field: {central_field}")
            central_phase_center = self.uvdata[band_name][central_field]['phase_center']
        
        # Collect all phase-shifted data
        all_uvreals = []
        all_uvimags = []
        all_uvdist = []
        all_uvwghts = []
        
        for field_name, field_data in self.uvdata[band_name].items():
            if field_name == 'metadata':
                continue
                
            # Calculate phase shift needed
            field_phase_center = field_data['phase_center']
            dRA = (central_phase_center[0] - field_phase_center[0]) * 3600  # to arcsec
            dDec = (central_phase_center[1] - field_phase_center[1]) * 3600  # to arcsec
            
            # Apply phase shift
            shifted_data = self.apply_phase_shift(dRA, dDec, field_data)
            
            # Collect data from all SPWs in this field
            for spw_name, spw_data in shifted_data.items():
                if spw_name == 'phase_center':
                    continue
                    
                # Calculate UV distance
                uvdist = np.sqrt(spw_data.uwave**2 + spw_data.vwave**2)
                
                all_uvreals.append(spw_data.uvreal)
                all_uvimags.append(spw_data.uvimag)
                all_uvdist.append(uvdist)
                all_uvwghts.append(spw_data.suvwght)
        
        # Concatenate all data
        UVreals = np.concatenate(all_uvreals)
        UVimags = np.concatenate(all_uvimags)
        UVdist = np.concatenate(all_uvdist)
        UVwghts = np.concatenate(all_uvwghts)
        
        # Perform binning
        bins = np.linspace(0, len(UVdist[UVdist > 0]) - 1, nbins, dtype=int)
        UVdist_sorted = np.sort(UVdist[UVdist > 0])
            
        UVrealbinned = np.empty(nbins - 1)
        UVrealerrors = np.empty(nbins - 1)
        UVimagbinned = np.empty(nbins - 1)
        UVimagerrors = np.empty(nbins - 1)
        bin_cs = np.empty(nbins - 1)
            
        for i in range(len(bins) - 1):
            mask = (UVdist > UVdist_sorted[bins[i]]) & (UVdist <= UVdist_sorted[bins[i + 1]])        
            
            UVrealbinned[i], UVrealerrors[i] = np.average(UVreals[mask], weights=UVwghts[mask], returned=True) 
            UVimagbinned[i], UVimagerrors[i] = np.average(UVimags[mask], weights=UVwghts[mask], returned=True)

            UVrealerrors[i] = UVrealerrors[i]**(-0.5) 
            UVimagerrors[i] = UVimagerrors[i]**(-0.5) 
                
            bin_cs[i] = np.median(UVdist[mask])

        return UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, UVdist_sorted[bins], bin_cs

    def plot_radial_distributions(self, nbins=20, save_plots=True, output_dir='../plots/uvplots/', 
                                custom_phase_center=None, use_style=True, data_name: str | None = None, **kwargs):
        """
        Plot radial UV distributions.

        If data_name is provided, only that dataset is plotted; otherwise all datasets.
        """
        # Setup plot style if requested
        if use_style:
            style_applied = setup_plot_style()
            
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Select datasets
        if data_name is not None:
            if data_name not in self.uvdata:
                raise ValueError(f"Dataset '{data_name}' not found. Available: {[k for k in self.uvdata if k!='metadata']}")
            dataset_names = [data_name]
        else:
            dataset_names = [name for name in self.uvdata if name != 'metadata']

        # Handle nbins as single value or list
        if isinstance(nbins, (int, float)):
            nbins_list = [int(nbins)] * len(dataset_names)
        elif isinstance(nbins, (list, tuple)):
            if len(nbins) != len(dataset_names):
                raise ValueError(f"Length of nbins list ({len(nbins)}) must match number of datasets ({len(dataset_names)})")
            nbins_list = [int(n) for n in nbins]
        else:
            raise ValueError("nbins must be an integer or a list of integers")

        fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.0})
        fig.set_figwidth(8)
        fig.set_figheight(6)
        
        # Initialize color counter
        color_idx = 0
        
        for i, name in enumerate(dataset_names):
            # Get the appropriate nbins for this dataset
            current_nbins = nbins_list[i]
            
            # Get binned data with optional custom phase center
            UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, bin_edges, bin_centers = self.get_binned_uvdatapoints(
                name, current_nbins, custom_phase_center
            )
            
            # Create the plot with color and label
            self._plot_single_radial_distribution(
                UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, 
                bin_edges, bin_centers, name, save_plots, output_dir, axes, color_idx, **kwargs
            )
            color_idx += 1
        
        plt.tight_layout()
        
        # Add legends to both panels
        axes[0].legend(frameon=True, loc='lower right')
        
        if save_plots:
            filename = 'UVradial_data_combined.png' if data_name is None else f'UVradial_{data_name}.png'
            plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_dir}/{filename}")
        
        plt.show()


    
    def _plot_single_radial_distribution(self, UVrealbinned, UVrealerrors, UVimagbinned, UVimagerrors, 
                                       bin_edges, bin_centers, name, save_plots, output_dir, axes, color_idx, **kwargs):
        """
        Create a single radial distribution plot.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to matplotlib errorbar functions.
        """

        # Define colors for different datasets
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        color = colors[color_idx % len(colors)]

        # Convert bin edges to k-lambda units (assuming they're already in the right units)
        x = bin_centers / 1e3  # Convert to k-lambda if needed
        
        # Real part plot
        ax = axes[0]
        
        # Prepare errorbar kwargs, preserving required styling
        errorbar_kwargs = kwargs.copy()
        errorbar_kwargs.update({
            'ls': '',
            'c': color,
            'markeredgecolor': color,
            'markerfacecolor': 'white',
            'marker': 'D',
            'label': f'{name} (Real)',
            'alpha': errorbar_kwargs.get('alpha', 0.8)
        })
        
        # Handle markersize - use from kwargs if provided
        if 'markersize' not in errorbar_kwargs:
            errorbar_kwargs['markersize'] = 12
        
        ax.errorbar(x, UVrealbinned * 1e3, # Convert to mJy
                   xerr=[abs(bin_edges[:-1]/1e3 - x), abs(bin_edges[1:]/1e3 - x)], 
                   yerr=UVrealerrors * 1e3, 
                   **errorbar_kwargs)

        ax.axhline(0, ls='dashed', c='gray', alpha=0.7)
        ax.set_ylabel('Re(V) [mJy]')
        ax.set_xlabel('')
        
        # Add secondary axis for spatial scale (only once)
        if color_idx == 0:
            from ..utils.utils import uvdist_to_arcsec, arcsec_to_uvdist
            secax = ax.secondary_xaxis('top', functions=(uvdist_to_arcsec, arcsec_to_uvdist))
            secax.set_xlabel('Spatial scale ["]')
            ax.tick_params(axis='x', which='both', top=False)
        
        # Set axis limits
        ax.set_ylim(-2.1, 0.6)
        ax.set_xlim(1e0,1e1)

        ax.set_xscale('log')

        # Imaginary part plot
        ax = axes[1]
        
        # Prepare errorbar kwargs for imaginary part, preserving required styling
        errorbar_kwargs_imag = kwargs.copy()
        errorbar_kwargs_imag.update({
            'ls': '',
            'c': color,
            'markeredgecolor': color,
            'markerfacecolor': 'white',
            'marker': 'o',
            'label': f'{name} (Imag)',
            'alpha': errorbar_kwargs_imag.get('alpha', 0.8)
        })
        
        # Handle markersize - use from kwargs if provided
        if 'markersize' not in errorbar_kwargs_imag:
            errorbar_kwargs_imag['markersize'] = 10
        
        ax.errorbar(x, UVimagbinned * 1e3, # Convert to mJy
                   xerr=[abs(bin_edges[:-1]/1e3 - x), abs(bin_edges[1:]/1e3 - x)], 
                   yerr=UVimagerrors * 1e3, 
                   **errorbar_kwargs_imag)

        ax.axhline(0, ls='dashed', c='gray', alpha=0.7)
        ax.set_ylim(-0.4, 0.4)
        ax.set_xlim(1e0,1e1)

        ax.set_ylabel('Imag(V) [mJy]')
        ax.set_xlabel(r'uv distance [k$\lambda$]')
        ax.set_xscale('log')

        # Add target name if available (only once)
        if hasattr(self, 'target') and self.target and color_idx == 0:
            ax.text(0.03, 0.97, f'{self.target}', transform=axes[0].transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))