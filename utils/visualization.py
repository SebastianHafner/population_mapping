import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import numpy as np
from pathlib import Path
from utils import paths, geofiles


def plot_vhr(ax, city: str, patch_id: str, i: int = None, j: int = None, vis: str = 'true_color',
             scale_factor: float = 0.4):
    dirs = paths.load_paths()
    file = Path(dirs.DATASET) / 'satellite_data' / city / f'vhr_{city}_{patch_id}.tif'
    img, _, _ = geofiles.read_tif(file)
    if i is not None and j is not None:
        img = img[i:i+200, j:j+200]
    band_indices = [0, 1, 2] if vis == 'true_color' else [3, 2, 1]
    bands = img[:, :, band_indices] / scale_factor
    bands = bands.clip(0, 1)
    ax.imshow(bands)
    ax.set_axis_off()


def plot_blackwhite(ax, img: np.ndarray):
    ax.imshow(img.clip(0, 1), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_stable_buildings(ax, file_all: Path, file_stable: Path, show_title: bool = False):
    img_all, _, _ = read_tif(file_all)
    img_all = img_all > 0
    img_all = img_all if len(img_all.shape) == 2 else img_all[:, :, 0]

    img_stable, _, _ = read_tif(file_stable)
    img_stable = img_stable > 0
    img_stable = img_stable if len(img_stable.shape) == 2 else img_stable[:, :, 0]

    img_instable = np.logical_and(img_all, np.logical_not(img_stable)) * 2

    cmap = colors.ListedColormap(['white', 'red', 'blue'])
    boundaries = [0, 0.5, 1, 1.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(img_all + img_instable, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('ground truth')


def plot_stable_buildings_v2(ax, arr: np.ndarray, show_title: bool = True):
    cmap = colors.ListedColormap(['white', 'blue', 'red'])
    print(np.min(arr), np.max(arr))
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('ground truth')


def plot_probability(ax, probability: np.ndarray, title: str = None):
    # ax.imshow(probability, cmap='bwr', vmin=0, vmax=1)
    cmap = colors.ListedColormap(['blue', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    # ax.imshow(probability, cmap=cmap, norm=norm)
    # ax.imshow(probability, cmap='Reds', vmin=0, vmax=1.2)
    ax.imshow(probability, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_prediction(ax, prediction: np.ndarray, show_title: bool = False):
    cmap = colors.ListedColormap(['white', 'red'])
    boundaries = [0, 0.5, 1]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(prediction, cmap=cmap, norm=norm)
    # ax.imshow(prediction, cmap='Reds')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_title:
        ax.set_title('prediction')


def plot_probability_histogram(ax, probability: np.ndarray, show_title: bool = False):

    bin_edges = np.linspace(0, 1, 21)
    values = probability.flatten()
    ax.hist(values, bins=bin_edges, range=(0, 1))
    ax.set_xlim((0, 1))
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yscale('log')

    if show_title:
        ax.set_title('probability histogram')


if __name__ == '__main__':
    arr = np.array([[0, 0.01, 0.1, 0.89, 0.9, 1, 1, 1]]).flatten()
    # hist, bin_edges = np.histogram(arr, bins=10, range=(0, 1))
    cmap = mpl.cm.get_cmap('Reds')
    norm = mpl.colors.Normalize(vmin=0, vmax=1.2)

    rgba = cmap(norm(0))
    print(mpl.colors.to_hex(rgba))
    rgba = cmap(norm(1))
    print(mpl.colors.to_hex(rgba))
