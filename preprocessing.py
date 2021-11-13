import numpy as np
from pathlib import Path
from affine import Affine
import geopandas as gpd
from tqdm import tqdm
from utils import geofiles, paths


# split a feature (e.g. vhr satellite data) into patches according to extend of pop file and patch size
# also outputs a metadata file for the tiling
def tile_image(city: str, feature: str):
    dirs = paths.load_paths()
    output_dir = Path(dirs.DATASET) / 'features' / city / feature
    output_dir.mkdir(exist_ok=True)

    # getting the dimensions of the population file first
    pop_file = Path(dirs.RAW_DATA) / f'pop_{city}.tif'
    arr_pop, transform_pop, crs_pop = geofiles.read_tif(pop_file)
    height_pop, width_pop, _ = arr_pop.shape
    x_res_pop, _, x_min_pop, _, y_res_pop, y_max_pop, *_ = transform_pop

    file = Path(dirs.RAW_DATA) / f'{feature}_{city}.tif'
    arr, transform, crs = geofiles.read_tif(file)
    height, width, _ = arr.shape
    x_res, _, x_min, _, y_res, y_max, *_ = transform

    # cropping feature file to extend of population file
    i_start = int(abs((y_max - y_max_pop) / y_res))
    i_end = int(i_start + abs((height_pop * y_res_pop / y_res)))
    j_start = int((x_min_pop - x_min) // x_res)
    j_end = int(j_start + (width_pop * x_res_pop / x_res))

    arr = arr[i_start:i_end, j_start:j_end, ]
    height, width, _ = arr.shape
    patch_size = int(x_res_pop / x_res)

    n_rows, n_cols = height // patch_size, width // patch_size

    tiling_data = {
        'feature': feature,
        'patch_size': patch_size,  # size of patch in pixels
    }

    # earth engine output is row col
    assert(n_rows == height_pop and n_cols == width_pop)
    for i in tqdm(range(n_rows)):
        for j in range(n_cols):
            i_start, i_end = i * patch_size, (i + 1) * patch_size
            j_start, j_end = j * patch_size, (j + 1) * patch_size
            patch = arr[i_start:i_end, j_start:j_end, ]

            x_min_patch = x_min_pop + j_start * x_res
            y_max_patch = y_max_pop + i_start * y_res
            transform_patch = Affine(x_res, 0, x_min_patch, 0, y_res, y_max_patch)

            patch_id = f'{i:03d}-{j:03d}'
            file = output_dir / f'{feature}_{city}_{patch_id}.tif'
            geofiles.write_tif(file, patch, transform_patch, crs)

    tiling_data_file = Path(dirs.DATASET) / f'tiling_data_{feature}_{city}.json'
    geofiles.write_json(tiling_data_file, tiling_data)


def assemble_metadata(city: str, features: list):
    dirs = paths.load_paths()

    # getting the dimensions of the population file first
    pop_file = Path(dirs.RAW_DATA) / f'pop_{city}.tif'
    arr_pop, *_ = geofiles.read_tif(pop_file)

    # training-test split according to census units
    train_file = Path(dirs.RAW_DATA) / f'train_polygons_{city}.tif'
    train_area, _, _ = geofiles.read_tif(train_file)
    test_file = Path(dirs.RAW_DATA) / f'test_polygons_{city}.tif'
    test_area, _, _ = geofiles.read_tif(test_file)

    # polygons of census areas that are fully covered by population data
    # TODO: this layer could probably also be computed on the fly
    assessment_file = Path(dirs.RAW_DATA) / f'valid_polygons_{city}.tif'
    assessment_area, _, _ = geofiles.read_tif(assessment_file)

    def get_tiling_data_feature(feature: str) -> dict:
        tiling_data_file = Path(dirs.DATASET) / f'tiling_data_{feature}_{city}.json'
        tiling_data = geofiles.load_json(tiling_data_file)
        return tiling_data
    features_tiling_data = [get_tiling_data_feature(f) for f in features]

    metadata = {
        'features': features,
        'samples': [],  # list with all the samples
    }
    for feature, tiling_data in zip(features, features_tiling_data):
        metadata[f'{feature}_patch_size'] = tiling_data['patch_size']  # size of patch in pixels

    for index, pop in np.ndenumerate(arr_pop):
        i, j, _ = index

        sample = {
            'city': city,
            'population': pop,
            'i': i,
            'j': j,
            'train_poly': int(train_area[i, j, 0]),
            'test_poly': int(test_area[i, j, 0]),
            'valid_for_assessment': int(assessment_area[i, j, 0]),
        }

        for feature, tiling_data in zip(features, features_tiling_data):
            patch_id = f'{i:03d}-{j:03d}'
            patch_file = Path(dirs.DATASET) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
            assert(patch_file.exists())

        metadata['samples'].append(sample)

    metadata_file = Path(dirs.DATASET) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


def create_census_geojson(city: str, run_type: str):
    dirs = paths.load_paths()
    file = Path(dirs.RAW_DATA) / f'{run_type}ing_polygons.shp'
    gdf = gpd.read_file(file)
    gdf['poly_id'] = gdf['cluster'] + 1
    gdf = gdf[['POPULATION', 'poly_id', 'geometry']]
    out_file = Path(dirs.DATASET) / 'census_data' / f'{run_type}_polygons_{city}.geojson'
    gdf.to_file(out_file, driver='GeoJSON')


if __name__ == '__main__':
    # run_preprocessing('dakar', 1000)
    # for rt in ['train', 'test']:
    #     create_census_geojson('dakar', rt)
    # tile_image('dakar', 'vhr')
    # tile_image('dakar', 's2')
    # tile_image('dakar', 'bf')
    assemble_metadata('dakar', ['vhr', 's2', 'bf'])


