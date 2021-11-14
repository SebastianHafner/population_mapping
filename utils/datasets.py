import torch
from torchvision import transforms
from pathlib import Path
from abc import abstractmethod
import affine
import math
import numpy as np
import cv2
from utils import augmentations, geofiles


class AbstractPopulationMappingDataset(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.root_path = Path(cfg.PATHS.DATASET)

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