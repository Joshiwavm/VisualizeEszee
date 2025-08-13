from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import copy

from ..utils.utils import *
from ..utils.style import setup_plot_style

class PlotFourierSensitivity:
    """
    A class for visualizing Fourier mode sensitivity of ALMA AND ACT DATA
    """
    def getWeightDistribution(self, name, bins=np.logspace(np.log10(0.1), np.log10(150), 31)):
        """
        Returns binned sensitivity for a given uvdata set name.
        """
        uvdata = copy.deepcopy(self.uvdata[name])
        all_uvdists = []
        all_wgts = []
        
        for field in uvdata:
            if not field.startswith('field'): 
                continue
            for spw in uvdata[field]:
                if not spw.startswith('spw'): 
                    continue
                uwave = uvdata[field][spw].uwave
                vwave = uvdata[field][spw].vwave
                suvwght = uvdata[field][spw].suvwght
                uvdist = (uwave**2 + vwave**2)**0.5 / 1e3
                all_uvdists.append(uvdist)
                all_wgts.append(suvwght)
        
        all_uvdists = np.concatenate(all_uvdists)
        all_wgts = np.concatenate(all_wgts)
        std_binned = []
        
        for i in range(len(bins) - 1):
            mask = (all_uvdists > bins[i]) & (all_uvdists <= bins[i + 1])
            wgts_sum = np.nansum(all_wgts[mask])
            std_binned.append((1 / wgts_sum) ** 0.5 if wgts_sum > 0 else np.nan)
        
        bin_centers = 10 ** ((np.log10(bins[1:]) + np.log10(bins[:-1])) / 2)
        return np.array(bin_centers), np.array(std_binned)


    def getACTSensitivity(self):
        """
        Compute inverse-variance weighted sensitivities for ACT data.
        
        Returns
        -------
        dict
            Dictionary with sensitivity values for each frequency and dataset
        """
        act_sensitivities = {}
        
        for name in self.actdata:
            act_sensitivities[name] = {}
            
            # Compute 90 GHz sensitivity
            if '090' in self.actdata[name]:
                stds_090 = []
                for pa in self.actdata[name]['090']:
                    std = self.actdata[name]['090'][pa].std
                    if std > 0:
                        stds_090.append(std)
                
                if stds_090:  # Only compute if we have valid data
                    act_sensitivities[name]['090'] = np.sqrt(1 / np.nansum([1/s**2 for s in stds_090]))
                else:
                    act_sensitivities[name]['090'] = np.nan
            
            # Compute 150 GHz sensitivity
            if '150' in self.actdata[name]:
                stds_150 = []
                for pa in self.actdata[name]['150']:
                    std = self.actdata[name]['150'][pa].std
                    if std > 0:
                        stds_150.append(std)
                
                if stds_150:  # Only compute if we have valid data
                    act_sensitivities[name]['150'] = np.sqrt(1 / np.nansum([1/s**2 for s in stds_150]))
                else:
                    act_sensitivities[name]['150'] = np.nan
        
        return act_sensitivities

    def plot_weight_distributions(self, use_style=True, **kwargs):
        """
        Plots binned and point source sensitivity for all loaded uvdata sets.
        
        Parameters
        ----------
        use_style : bool, optional
            Whether to apply custom thesis style (default: True)
        **kwargs : dict
            Additional keyword arguments passed to matplotlib plotting functions.
            Common options include:
            - linewidth : float, line width for plots
            - alpha : float, transparency
            - linestyle : str, line style
            - marker : str, marker style
            - markersize : float, marker size
        """
        # Setup plot style if requested
        if use_style:
            style_applied = setup_plot_style()
        
        # Extract kwargs for different plot types
        plot_kwargs = kwargs.copy()
        axhline_kwargs = kwargs.copy()
        
        # Handle specific kwargs for axhline (dashed lines)
        if 'linestyle' in axhline_kwargs:
            axhline_kwargs['ls'] = axhline_kwargs.pop('linestyle')
        if 'ls' not in axhline_kwargs:
            axhline_kwargs['ls'] = '--'
        
        fig, ax = plt.subplots(constrained_layout=True)
        
        for i, name in enumerate(self.uvdata):
            bin_centers, std_binned = self.getWeightDistribution(name)
            ax.plot(bin_centers, std_binned * 1e6, c=f'C{i}', label=name, **plot_kwargs)
            ps_sens = (1/np.nansum(1/std_binned**2))**0.5*1e6
            ax.axhline(ps_sens, c=f'C{i}', **axhline_kwargs)

        if hasattr(self, 'actdata') and self.actdata:
            # Get ACT sensitivities using the dedicated method
            act_sensitivities = self.getACTSensitivity()

            # Plot ACT lines if values are available
            color_offset = len(self.uvdata)
            for j, name in enumerate(act_sensitivities):
                if '150' in act_sensitivities[name] and not np.isnan(act_sensitivities[name]['150']):
                    ax.plot([0.2, arcsec_to_uvdist(1.4*60)], 
                           [act_sensitivities[name]['150']*1e6]*2, 
                           label=f'ACT 150 GHz ({name})', c=f'C{color_offset+j*2}', ls='-', **plot_kwargs)
                if '090' in act_sensitivities[name] and not np.isnan(act_sensitivities[name]['090']):
                    ax.plot([0.2, arcsec_to_uvdist(2.0*60)], 
                           [act_sensitivities[name]['090']*1e6]*2, 
                           label=f'ACT 90 GHz ({name})', c=f'C{color_offset+j*2+1}', ls='-', **plot_kwargs)

        ax.text(0.03, 0.97, f'{self.target}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        ax.set_xlabel(r'uv-distance [k$\lambda$]', fontsize=12)
        ax.set_ylabel(r'$\sigma$ [$\mu$Jy]', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='x', which='both', top=False)
        
        ax.axis(xmin=1e0, xmax=1e2)
        secax = ax.secondary_xaxis('top', functions=(arcsec_to_uvdist, uvdist_to_arcsec))
        secax.set_xlabel('Spatial scale ["]', fontsize=12)
        plt.legend(frameon=True, loc=1)        
        plt.show()
