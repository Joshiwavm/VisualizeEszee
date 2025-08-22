"""
Style utilities for matplotlib plots.
Handles loading custom styles with graceful fallback.
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_thesis_style():
    """
    Try to load the thesis matplotlib style.
    
    Returns
    -------
    bool
        True if style was successfully loaded, False otherwise
    """
    style_path = Path.home() / "Documents" / "Projects" / "plotting" / "thesis.mplstyle"
    
    try:
        if style_path.exists():
            # Try to use the style
            plt.style.use(str(style_path))
            print(f"✓ Loaded custom style: {style_path}")
            return True
        else:
            print(f"Style file not found: {style_path}")
            return False
    except Exception as e:
        print(f"Warning: Could not load custom style {style_path}: {e}")
        return False

def apply_publication_defaults():
    """
    Apply some publication-ready defaults when custom style isn't available.
    """
    try:
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'lines.linewidth': 1.5,
            'axes.linewidth': 1.0,
            'grid.alpha': 0.3,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        print("✓ Applied publication-ready default styling")
        return True
    except Exception as e:
        print(f"Warning: Could not apply default styling: {e}")
        return False

def setup_plot_style():
    """
    Setup plotting style with fallback hierarchy:
    1. Try to load thesis.mplstyle
    2. Fall back to publication defaults
    3. Use matplotlib defaults if all else fails
    
    Returns
    -------
    str
        Description of which style was applied
    """
    # Try custom style first
    if load_thesis_style():
        return "thesis_style"
    
    # Fall back to publication defaults
    if apply_publication_defaults():
        return "publication_defaults"
    
    # Ultimate fallback - use matplotlib defaults
    print("Using matplotlib default styling")
    return "matplotlib_defaults"
