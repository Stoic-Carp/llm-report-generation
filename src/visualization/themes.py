"""Visual styling and themes for plots."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from config.settings import get_settings


# Professional color palette for business reports - refined for better contrast and elegance
PROFESSIONAL_COLORS = [
    '#003667',  # BMW Blue / Deep Navy
    '#6F6F6F',  # BMW Grey / Dark Grey
    '#2E86AB',  # Steel Blue
    '#A23B72',  # Berry
    '#F18F01',  # Burnt Orange
    '#C73E1D',  # Brick Red
    '#6A994E',  # Sage Green
    '#BC4749',  # Muted Red
    '#06A77D',  # Emerald
    '#118AB2',  # Cerulean
]

# Color palette for categorical data
CATEGORICAL_PALETTE = sns.color_palette(PROFESSIONAL_COLORS)


def setup_plot_style():
    """Configure matplotlib and seaborn styling for professional reports.

    Uses settings from configuration to set up plot appearance with
    professional styling suitable for business reports.
    """
    settings = get_settings()

    # Use a clean, professional style foundation
    # seaborn-whitegrid is generally cleaner
    sns.set_theme(style="whitegrid")
    
    # Override with custom professional styling
    mpl.rcParams.update({
        # Figure settings
        'figure.figsize': settings.visualization.figsize,
        'figure.dpi': settings.visualization.dpi,
        'figure.autolayout': True,  # Helps with spacing
        
        # Saving settings
        'savefig.dpi': settings.visualization.dpi,
        'savefig.format': settings.visualization.format,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        
        # Font settings - use system sans-serifs that look clean
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlepad': 20,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#E0E0E0',
        'legend.fancybox': True,
        
        # Axes settings
        'axes.linewidth': 1.0,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.labelcolor': '#2c3e50',
        'axes.edgecolor': '#bdc3c7',
        
        # Grid settings - make it subtle
        'grid.alpha': 0.4,
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.color': '#E0E0E0',
        
        # Tick settings
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.color': '#505050',
        'ytick.color': '#505050',
        
        # Line settings
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.5,
        
        # Colors
        'axes.prop_cycle': mpl.cycler('color', PROFESSIONAL_COLORS),
    })
    
    # Set seaborn palette
    sns.set_palette(PROFESSIONAL_COLORS)
