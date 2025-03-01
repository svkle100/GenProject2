import torch
from torch.utils.data.sampler import Sampler

from data_utils.BrainDataset import BrainDataset


class BrainSampler(Sampler):
    def __init__(self, data, tile_size, map_type):
        self.data = data
        self.brains = self.data.get_brains()
        self.tile_size = tile_size
        self.brain_prob = torch.tensor([1 for brain in self.brains], dtype=torch.float32)
        self.map_type = map_type


    def __iter__(self):
        while True:
            brain = self.brains[torch.multinomial(self.brain_prob, 1)]
            row = torch.randint(self.data.get_shape(brain)[0]-self.tile_size, (1,)).item()
            column = torch.randint(self.data.get_shape(brain)[1]-self.tile_size, (1,)).item()
            yield (brain[0], brain[1], brain[2],self.map_type, row, column, self.tile_size)

    def __len__(self):
        return float('inf')