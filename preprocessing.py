from pathlib import Path
from affine import Affine
from utils import geofiles, paths


# only works for southern hemisphere
def create_tiles(file: Path, output_dir: Path, tile_size: int):
    base_name = file.stem

    arr, transform, crs = geofiles.read_tif(file)
    height, width, _ = arr.shape
    n_rows, n_cols = height // tile_size, width // tile_size
    img_res_x, _, img_min_x, _, img_res_y, img_max_y, *_ = transform

    # earth engine output is row col
    for i in range(n_rows):
        for j in range(n_cols):
            i_start, i_end = i * tile_size, (i + 1) * tile_size
            j_start, j_end = j * tile_size, (j + 1) * tile_size
            tile_arr = arr[i_start:i_end, j_start:j_end, ]

            tile_min_x = img_min_x + j_start * img_res_x
            tile_max_y = img_max_y + i_start * img_res_y
            tile_transform = Affine(img_res_x, 0, tile_min_x, 0, img_res_y, tile_max_y)

            file = output_dir / f'{base_name}_{i_start:010d}-{j_start:010d}.tif'
            geofiles.write_tif(file, tile_arr, tile_transform, crs)


if __name__ == '__main__':
    dirs = paths.load_paths()
    output_folder = Path(dirs.DATASET) / 'satellite_data'
    satellite_file = Path(dirs.RAW_SATELLITE_FILE)
    create_tiles(satellite_file, output_folder, 1024)



