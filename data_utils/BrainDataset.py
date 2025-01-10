import json
import os

import h5py
import numpy as np
import torch
from imageio.core import image_as_uint
from torch.utils.data import Dataset

from . import get_brain_section
from .utils import get_dataset_path

class BrainDataset(Dataset):
    def __init__(self, brains, map_type, resolution="00"):
        self.files = {}
        if type(map_type) != list:
            map_type = [map_type]
        for brain, section, region in brains:
            for m in map_type:
                self.files[(brain, section, region, m)] = get_brain_section(brain, section, region, m)
        self.stats = {}
        for m in map_type:
            self.stats[m] = json.load(open(os.path.join(get_dataset_path(), m, 'stats.json')))
        self.resolution = resolution

    def __getitem__(self, index):
        brain, section, region, map_type, row, column, patch_size = index
        if type(map_type) != list:
            map_type = [map_type]
        output = []
        for m in map_type:
            brain_image = self.files[(brain, section, region, m)]["pyramid"][self.resolution][row:row+patch_size, column:column+patch_size]
            brain_image = torch.tensor(brain_image, dtype=torch.float32)
            brain_image = (brain_image - torch.tensor(self.stats[m]["mean"])) / torch.tensor(self.stats[m]["std"])
            if brain_image.ndim == 2:
                brain_image = brain_image.unsqueeze(2)
            output.append(brain_image.permute(2, 0, 1))
        return tuple(output)

    def get_brains(self):
        return list(self.files.keys())

    def get_shape(self, brain):
        return self.files[brain]["pyramid"][self.resolution].shape