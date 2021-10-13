from pathlib import Path
from affine import Affine
from tqdm import tqdm
from utils import geofiles, paths


# only works for southern hemisphere
def preprocess(city: str, tile_size: int):
    dirs = paths.load_paths()
    output_dir = Path(dirs.DATASET) / 'satellite_data' / city
    output_dir.mkdir(exist_ok=True)
    satellite_file = Path(dirs.RAW_SATELLITE_DATA) / f'satellite_data_{city}.tif'

    arr, transform, crs = geofiles.read_tif(satellite_file)
    height, width, _ = arr.shape
    n_rows, n_cols = height // tile_size, width // tile_size
    img_res_x, _, img_min_x, _, img_res_y, img_max_y, *_ = transform

    metadata = {'tile_size': tile_size, 'samples': []}

    # earth engine output is row col
    for i in tqdm(range(n_rows)):
        for j in range(n_cols):
            i_start, i_end = i * tile_size, (i + 1) * tile_size
            j_start, j_end = j * tile_size, (j + 1) * tile_size
            tile_arr = arr[i_start:i_end, j_start:j_end, ]

            tile_min_x = img_min_x + j_start * img_res_x
            tile_max_y = img_max_y + i_start * img_res_y
            tile_transform = Affine(img_res_x, 0, tile_min_x, 0, img_res_y, tile_max_y)

            patch_id = f'{i_start:010d}-{j_start:010d}'
            file = output_dir / f'satellite_data_{city}_{patch_id}.tif'
            geofiles.write_tif(file, tile_arr, tile_transform, crs)

            metadata['samples'].append({
                'city': city,
                'patch_id': patch_id,
            })

    metadata_file = output_dir.parent / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


if __name__ == '__main__':
    preprocess('dakar', 2048)



