import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import affine
import math
import numpy as np
import cv2
from utils import augmentations, geofiles, paths


class AbstractPopulationMappingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dirs = paths.load_paths()
        self.root_path = Path(dirs.DATASET)

        self.features = cfg.DATALOADER.FEATURES
        self.patch_size = cfg.DATALOADER.PATCH_SIZE
        self.n_channels = cfg.MODEL.IN_CHANNELS

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    # generic data loading function used for different features (e.g. vhr satellite data)
    def _get_patch_data(self, feature: str, city: str, i: int, j: int) -> np.ndarray:
        file = self.root_path / 'features' / city / feature / f'{feature}_{city}_{i:03d}-{j:03d}.tif'
        img, _, _ = geofiles.read_tif(file)

        if feature == 'vhr':
            band_indices = self.cfg.DATALOADER.VHR_BAND_INDICES
        elif feature == 's2':
            band_indices = self.cfg.DATALOADER.S2_BAND_INDICES
        else:
            band_indices = [0]
        img = img[:, :, band_indices]

        if feature == 'vhr':
            img = np.clip(img / self.cfg.DATALOADER.VHR_MAX_REFLECTANCE, 0, 1)

        # resampling images to desired patch size
        if img.shape[0] != self.patch_size or img.shape[1] != self.patch_size:
                img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)

        return np.nan_to_num(img).astype(np.float32)

    # loading patch data for all features
    def _get_patch_net_input(self, city: str, i: int, j: int) -> np.ndarray:
        patch_net_input = np.empty((self.patch_size, self.patch_size, self.n_channels), dtype=np.single)
        start_i = 0
        for feature in self.features:
            feature_patch_data = self._get_patch_data(feature, city, i, j)
            feature_n_channels = feature_patch_data.shape[-1]
            patch_net_input[:, :, start_i:start_i + feature_n_channels] = feature_patch_data
            start_i += feature_n_channels
        return patch_net_input

    @staticmethod
    def pop_log_conversion(pop: float) -> float:
        if pop == 0:
            return 0
        else:
            return math.log10(pop)


# dataset for urban extraction with building footprints
class CellPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, cfg, run_type: str, no_augmentations: bool = False, include_unlabeled: bool = True):
        super().__init__(cfg)

        self.run_type = run_type
        self.no_augmentations = no_augmentations
        self.include_unlabeled = include_unlabeled

        if cfg.DATASET.CITY_SPLIT:
            self.cities = list(cfg.DATASET.TRAINING) if run_type == 'train' else list(cfg.DATASET.TEST)
        else:
            self.cities = list(cfg.DATASET.LABELED_CITIES)
        if include_unlabeled and cfg.DATALOADER.INCLUDE_UNLABELED:
            self.cities += cfg.DATASET.UNLABELED_CITIES

        self.samples = []
        for city in self.cities:
            city_metadata_file = self.root_path / f'metadata_{city}.json'
            city_metadata = geofiles.load_json(city_metadata_file)
            self.samples.extend(city_metadata['samples'])

        # removing nan values
        self.samples = [s for s in self.samples if not math.isnan(s['population'])]

        if not cfg.DATASET.CITY_SPLIT:
            self.samples = [s for s in self.samples if s[f'{run_type}_poly'] != 0]

        if no_augmentations:
            self.transform = transforms.Compose([augmentations.Numpy2Torch()])
        else:
            self.transform = augmentations.compose_transformations(cfg)

        self.length = len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        if self.cfg.DATALOADER.LOG_POP:
            population = self.pop_log_conversion(float(sample['population']))
        else:
            population = float(sample['population'])

        patch_data = self._get_patch_net_input(city, i, j)
        x = self.transform(patch_data)

        item = {
            'x': x,
            'y': torch.tensor([population]),
            'i': i,
            'j': j,
            'train_poly': sample['train_poly'],
            'test_poly': sample['test_poly'],
            'valid_for_assessment': sample['valid_for_assessment'],
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for urban extraction with building footprints
class CensusPopulationDataset(AbstractPopulationMappingDataset):

    def __init__(self, cfg, city: str, run_type: str, poly_id: int):
        super().__init__(cfg)

        self.run_type = run_type

        metadata_file = self.root_path / f'metadata_{city}.json'
        metadata = geofiles.load_json(metadata_file)
        all_samples = metadata['samples']
        self.samples = [s for s in all_samples if not math.isnan(s['population']) and s[f'{run_type}_poly'] == poly_id]
        self.valid_for_assessment = True if np.all([[s['valid_for_assessment'] for s in self.samples]]) else False

        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        self.length = len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        city = sample['city']
        i, j = sample['i'], sample['j']
        population = float(sample['population'])

        patch_data = self._get_patch_net_input(city, i, j)
        x = self.transform(patch_data)

        item = {
            'x': x,
            'y': torch.tensor([population]),
            'i': i,
            'j': j,
            'train_poly': sample['train_poly'],
            'test_poly': sample['test_poly'],
            'valid_for_assessment': sample['valid_for_assessment'],
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.cities)} sites.'


# dataset for classifying a scene
class TilesInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, site: str):
        super().__init__()

        self.cfg = cfg
        self.site = site

        dirs = paths.load_paths()
        self.root_dir = Path(dirs.DATASET)
        self.transform = transforms.Compose([augmentations.Numpy2Torch()])

        # getting all files
        samples_file = self.root_dir / site / 'samples.json'
        metadata = geofiles.load_json(samples_file)
        self.samples = metadata['samples']
        self.length = len(self.samples)

        self.patch_size = metadata['patch_size']

        # computing extent
        patch_ids = [s['patch_id'] for s in self.samples]
        self.coords = [[int(c) for c in patch_id.split('-')] for patch_id in patch_ids]
        self.max_y = max([c[0] for c in self.coords])
        self.max_x = max([c[1] for c in self.coords])

        # creating boolean feature vector to subset sentinel 1 and sentinel 2 bands
        self.s1_indices = self._get_indices(metadata['sentinel1_features'], cfg.DATALOADER.SENTINEL1_BANDS)
        self.s2_indices = self._get_indices(metadata['sentinel2_features'], cfg.DATALOADER.SENTINEL2_BANDS)
        if cfg.DATALOADER.MODE == 'sar':
            self.n_features = len(self.s1_indices)
        elif cfg.DATALOADER.MODE == 'optical':
            self.n_features = len(self.s2_indices)
        else:
            self.n_features = len(self.s1_indices) + len(self.s2_indices)

    def __getitem__(self, index):

        # loading metadata of sample
        sample = self.samples[index]
        patch_id_center = sample['patch_id']

        y_center, x_center = patch_id_center.split('-')
        y_center, x_center = int(y_center), int(x_center)

        extended_patch = np.zeros((3 * self.patch_size, 3 * self.patch_size, self.n_features), dtype=np.float32)

        for i in range(3):
            for j in range(3):
                y = y_center + (i - 1) * self.patch_size
                x = x_center + (j - 1) * self.patch_size
                patch_id = f'{y:010d}-{x:010d}'
                if self._is_valid_patch_id(patch_id):
                    patch = self._load_patch(patch_id)
                else:
                    patch = np.zeros((self.patch_size, self.patch_size, self.n_features), dtype=np.float32)
                i_start = i * self.patch_size
                i_end = (i + 1) * self.patch_size
                j_start = j * self.patch_size
                j_end = (j + 1) * self.patch_size
                extended_patch[i_start:i_end, j_start:j_end, :] = patch

        if sample['is_labeled']:
            label, _, _ = self._get_label_data(patch_id_center)
        else:
            # dummy_label = np.zeros((extended_patch.shape[0], extended_patch.shape[1], 1), dtype=np.float32)
            dummy_label = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
            label = dummy_label
        extended_patch, label = self.transform((extended_patch, label))

        item = {
            'x': extended_patch,
            'y': label,
            'i': y_center,
            'j': x_center,
            'site': self.site,
            'patch_id': patch_id_center,
            'is_labeled': sample['is_labeled']
        }

        return item

    def _is_valid_patch_id(self, patch_id):
        patch_ids = [s['patch_id'] for s in self.samples]
        return True if patch_id in patch_ids else False

    def _load_patch(self, patch_id):
        mode = self.cfg.DATALOADER.MODE
        if mode == 'optical':
            img, _, _ = self._get_sentinel2_data(patch_id)
        elif mode == 'sar':
            img, _, _ = self._get_sentinel1_data(patch_id)
        else:  # fusion baby!!!
            s1_img, _, _ = self._get_sentinel1_data(patch_id)
            s2_img, _, _ = self._get_sentinel2_data(patch_id)
            img = np.concatenate([s1_img, s2_img], axis=-1)
        return img

    def _get_sentinel1_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel1' / f'sentinel1_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s1_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_sentinel2_data(self, patch_id):
        file = self.root_dir / self.site / 'sentinel2' / f'sentinel2_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(file)
        img = img[:, :, self.s2_indices]
        return np.nan_to_num(img).astype(np.float32), transform, crs

    def _get_label_data(self, patch_id):
        label = self.cfg.DATALOADER.LABEL
        threshold = self.cfg.DATALOADER.LABEL_THRESH

        label_file = self.root_dir / self.site / label / f'{label}_{self.site}_{patch_id}.tif'
        img, transform, crs = geofiles.read_tif(label_file)
        if threshold >= 0:
            img = img > threshold

        return np.nan_to_num(img).astype(np.float32), transform, crs

    def get_arr(self, dtype=np.uint8):
        height = self.max_y + self.patch_size
        width = self.max_x + self.patch_size
        return np.zeros((height, width, 1), dtype=dtype)

    def get_geo(self):
        patch_id = f'{0:010d}-{0:010d}'
        # in training and validation set patches with no BUA were not downloaded -> top left patch may not be available
        if self._is_valid_patch_id(patch_id):
            _, transform, crs = self._get_sentinel1_data(patch_id)
        else:
            # use first patch and covert transform to that of uupper left patch
            patch = self.samples[0]
            patch_id = patch['patch_id']
            i, j = patch_id.split('-')
            i, j = int(i), int(j)
            _, transform, crs = self._get_sentinel1_data(patch_id)
            x_spacing, x_whatever, x_start, y_whatever, y_spacing, y_start, *_ = transform
            x_start -= (x_spacing * j)
            y_start -= (y_spacing * i)
            transform = affine.Affine(x_spacing, x_whatever, x_start, y_whatever, y_spacing, y_start)
        return transform, crs

    @staticmethod
    def _get_indices(bands, selection):
        return [bands.index(band) for band in selection]

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples across {len(self.sites)} sites.'
