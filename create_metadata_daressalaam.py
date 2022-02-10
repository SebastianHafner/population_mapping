import numpy as np
from pathlib import Path
from utils import geofiles, parsers
import geopandas as gpd


# -f s2 bf -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
def assemble_metadata(features: list, raw_data_path: str, dataset_path: str):
    city = 'daressalaam'
    metadata = {
        'features': features,
        'census': {},
        'split': {},
        'samples': [],  # list with all the samples
    }

    # rasterized version of census file with 'cat' as pixel value (0 is no census)
    units_file = Path(raw_data_path) / city / f'census_grid_cloudfree.tif'
    units, _, _ = geofiles.read_tif(units_file)
    # pixel values are population for 100 m grid
    # pop_file = Path(raw_data_path) / city / f'pop_resampled.tiff'
    # pop, _, _ = geofiles.read_tif(pop_file)

    for index, unit in np.ndenumerate(units):
        i, j, _ = index
        patch_id = f'{i:03d}-{j:03d}'
        if unit == 0:
            for feature in features:
                patch_file = Path(dataset_path) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
                if patch_file.exists():
                    patch_file.unlink()
        else:
            sample = {
                'city': city,
                'population': -1,  # float(pop[index])
                'i': i,
                'j': j,
                'split': 'test',
                'unit': int(unit),
            }
            for feature in features:
                patch_file = Path(dataset_path) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
                assert (patch_file.exists())
            metadata['samples'].append(sample)

    census_file = Path(raw_data_path) / city / f'gridded_spatial_units_cloudfree.shp'
    gdf = gpd.read_file(census_file)
    for index, row in gdf.iterrows():
        unit_nr = row['cat']
        unit_pop = row['tza_pop__7']
        metadata['census'][unit_nr] = unit_pop
        metadata['split'][unit_nr] = 'test'

    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


if __name__ == '__main__':
    args = parsers.metadata_argument_parser().parse_known_args()[0]
    assemble_metadata(args.features, args.rawdata_dir, args.dataset_dir)