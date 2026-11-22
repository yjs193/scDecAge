"""Small display and export helpers for the notebooks."""
import base64
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display, HTML
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parents[2] / 'data'

def setup(display_fraction=.45, display_width=480):
    font = 'Arial' if any(item.name == 'Arial' for item in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({"font.family":font, "font.size":11, "axes.labelsize":11,
        "axes.titlesize":12, "xtick.labelsize":10, "ytick.labelsize":10,
        "legend.fontsize":10, "svg.fonttype":"none", "figure.facecolor":"white",
        "axes.facecolor":"white", "savefig.facecolor":"white", "axes.spines.top":False,
        "axes.spines.right":False, "axes.linewidth":.8, "grid.color":"#E7E9EC", "grid.linewidth":.6})
    display(HTML(f'<style>.jp-OutputArea-output img,.output_png img{{width:{display_fraction*100:g}% !important;max-width:{display_width}px;height:auto;}}</style>'))

def output(fig, directory, stem):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        fig.savefig(directory/(stem+'.'+ext),dpi=800,bbox_inches='tight',facecolor='white',pad_inches=.04)
    plt.close(fig)

def letter(ax, text, x=-.13):
    ax.text(x,1.06,text,transform=ax.transAxes,fontsize=19,fontweight='bold',va='bottom')

def show_figure(path, width=480):
    Image.MAX_IMAGE_PIXELS=200_000_000
    picture=Image.open(path).convert('RGB');picture.thumbnail((1100,1600),Image.Resampling.LANCZOS)
    buffer=BytesIO();picture.save(buffer,format='PNG')
    display({'image/png':base64.b64encode(buffer.getvalue()).decode('ascii'),'text/plain':'Analysis panel'},
            metadata={'image/png':{'width':width}},raw=True)

def analysis_directory(root, dataset, partition):
    base = Path(root) / dataset
    matches = [path for path in base.iterdir()
               if path.is_dir() and path.name.rsplit('_', 1)[-1].isdigit()
               and int(path.name.rsplit('_', 1)[-1]) == int(partition)]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one analysis input directory for {dataset}, index {partition}.')
    return matches[0]
