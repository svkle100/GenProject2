import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from IPython.core.pylabtools import figsize


def get_dataset_path():
    if os.path.exists('/home/sven/Documents/Uni/Gen/project2/data'):
        return '/home/sven/Documents/Uni/Gen/project2/data'
    else:
        raise FileNotFoundError('Please add your dataset path to utils.py')

def get_brain_section(brain, section, region, view):
    dir = os.listdir(os.path.join(get_dataset_path(), view))
    file = ""
    for f in dir:
        if f"Vervet{brain}_s{section:04d}_{region if region is not None else ""}" in f:
            file = f
    file = h5py.File(os.path.join(get_dataset_path(), f"{view}/{file}"), 'r')
    return file

def visualize_batch(batch, map_type, normalized=True):
    if type(batch) == torch.Tensor:
        batch = batch.clone().detach().cpu()
    if len(batch.shape) == 4 and type(batch) == torch.Tensor:
        batch = batch.permute(0, 2, 3, 1)
    if len(batch.shape) == 2 or (len(batch.shape) == 3 and batch.shape[-1] == 3):
        bs = 1
        batch = np.array([batch])
    else:
        bs = batch.shape[0]

    if normalized:
        batch = unnormalize(batch, map_type)
    fig, axs = plt.subplots(1, bs, figsize = [bs * 2 * 5, bs*5])
    if bs == 1:
        axs.axis('off')
        axs.imshow(batch[0])
    else:
        for i in range(bs):
            axs[i].axis('off')
            axs[i].imshow(batch[i])
    plt.show()


def unnormalize(batch, map_type):
    stats = json.load(open(os.path.join(get_dataset_path(), map_type, 'stats.json')))
    mean, std = np.array(stats["mean"]), np.array(stats["std"])
    batch = batch * std + mean
    batch = batch.clone().detach().cpu().numpy()
    batch = np.clip(batch, 0, 255).astype(np.uint8)
    return batch


def save_model(gen, disc, options):
    name = str(hash(gen.parameters()))
    dir = os.path.join(get_dataset_path(), "../models", name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, "options.json"), "w") as outfile:
        json.dump(options, outfile)
    torch.save(gen.state_dict(), os.path.join(dir, "model.pt"))
    torch.save(disc.state_dict(), os.path.join(dir, "disc.pt"))
