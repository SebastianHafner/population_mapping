import numpy as np
from pathlib import Path
from utils import geofiles, parsers
import numpy as np


def assemble_metadata(features: list, raw_data_path: str, dataset_path: str):
    city = 'ndjamena'
    metadata = {
        'features': features,
        'census': {},
        'split': {},
        'samples': [],  # list with all the samples
    }

    mask_file = Path(raw_data_path) / city / f'grid_mask_{city}.tif'
    grid, _, _ = geofiles.read_tif(mask_file)

    for index, cell_mask in np.ndenumerate(grid):
        i, j, _ = index
        patch_id = f'{i:03d}-{j:03d}'
        if cell_mask == 0:
            for feature in features:
                patch_file = Path(dataset_path) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
                if patch_file.exists():
                    patch_file.unlink()
        else:
            sample = {
                'city': city,
                'population': -1,
                'i': i,
                'j': j,
                'split': 'test',
            }
            for feature in features:
                patch_file = Path(dataset_path) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
                assert (patch_file.exists())
            metadata['samples'].append(sample)

    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


if __name__ == '__main__':
    args = parsers.metadata_argument_parser().parse_known_args()[0]
    assemble_metadata(args.features, args.rawdata_dir, args.dataset_dir)

