import os
from random import randint
import uuid
from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import tasks
from curriculum import Curriculum
from schema import schema
from models import build_model
import wandb
import pickle
import random
import numpy as np
import torch
import gc
import json
from torch.utils.data import DataLoader
from transformers import get_scheduler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True

def count_files_in_folder(folder, prefix, suffix):
    """
    Counts the number of files in dataset folder that matches prefix and suffix.

    Args:
        folder (str): The path to the folder where the files are located.
        prefix (str): The prefix that the file names must start with.
        suffix (str): The file extention that the file must end with.

    Returns:
        int: the number of files in the folder.
    """
    return len([f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(suffix)])

def load_dataset_full(pickle_folder):
    """
    Loads entire dataset from a specified folder

    Args:
        pickle_folder (str): The path to the folder containing the pickle files.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.

    Returns:
        list: A list containing the loaded data from all the pickle files in the folder.
    """
    dataset = []
    masses = []
    lengths = []
    dataset_full = []
    total_files = count_files_in_folder(pickle_folder, "multipendulum_", ".pkl")
    with tqdm(total=total_files, desc="Loading all files", leave=False) as load_pbar:
        for i in range(total_files):
            pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    # xs, ys = pickle.load(f)
                    xs, ys, mass, length = pickle.load(f)
                    dataset.append((xs, ys))
                    masses.append(mass)
                    lengths.append(length)
                    dataset_full.append((xs, ys, mass, length))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset, dataset_full, masses, lengths

def load_dataset_chunk(pickle_folder, start_idx, end_idx):
    """
    Loads chunk of dataset files from specified folder within a given index range, 
    and returns the data as a list of tuples.

    Args:
        pickle_folder (str): The path to folder containing the pickle files.
        start_idx (int): The starting index of the pickle files to load.
        end_idx (int): The ending index of the pickle files to load.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
    """
    dataset = []
    masses = []
    lengths = []
    dataset_full = []
    with tqdm(total=end_idx - start_idx + 1, desc=f"Loading files {start_idx}-{end_idx}", leave=False) as load_pbar:
        for i in range(start_idx, end_idx + 1):
            pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    # xs, ys = pickle.load(f)
                    xs, ys, mass, length = pickle.load(f)
                    dataset.append((xs, ys))
                    masses.append(mass)
                    lengths.append(length)
                    dataset_full.append((xs, ys, mass, length))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset, dataset_full, masses, lengths

def get_files_from_folder(folderpath):
    """ Get all .pkl files from a folder """
    return[os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith(".pkl")]

def find_nearest_neighbors(k, query, dataset, context=1, metric="euclidean"):
    """
    Finds the k-nearest neighbors to a given query point in a dataset.

    Args:
        k (int): The number of nearest neighbors to find.
        query (any): The query point to find the nearest neighbors to.
        dataset (list): A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
        context (int): The number of context points to use for the query.
        metric (str): The distance metric to use for finding the nearest neighbors.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
    """
    # distances = []
    # for xs, ys in dataset:
    #     if metric == "euclidean":
    #         distance = np.linalg.norm(np.array(query) - np.array(xs))
    #     distances.append((xs, ys, distance))
    # distances.sort(key=lambda x: x[2])

    # for xs, ys in dataset:
    #     if metric == "euclidean":
    #         distance = torch.norm(query - xs)
    #     distances.append((xs, ys, distance))
    # distances.sort(key=lambda x: x[2].item())
    # return distances[:k]
    combined_dataset = []
    combined_query = []
    for xs, ys in dataset:
        xs_b = torch.tensor(xs, dtype=torch.float32)
        ys_b = torch.tensor(ys, dtype=torch.float32)
        if context == 1:
            combined_dataset.append(xs_b)
        else:
            combined = combine(xs_b.unsqueeze(0), ys_b.unsqueeze(0)).squeeze(0)
            combined_dataset.append(combined)

    combined_dataset = torch.stack(combined_dataset)
    query = query.unsqueeze(0)

    for x,y in query:
        x_b = torch.tensor(x, dtype=torch.float32)
        y_b = torch.tensor(y, dtype=torch.float32)
        if context == 1:
            combined_query.append(x_b)
        else:
            combined = combine(x_b.unsqueeze(0), y_b.unsqueeze(0)).squeeze(0)
            combined_query.append(combined)

    if metric == "euclidean":
        distances = torch.norm(combined_dataset - query, p=2, dim=1)

    distances, indices = torch.topk(distances, k, largest=False, sorted=True)
    return indices, distances

def combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs


def main(args):
    # Load dataset
    dataset_folder = args.dataset_filesfolder
    pickle_folder = args.pickle_folder
    fullpicklepath = os.path.join(dataset_folder, pickle_folder)
    total_files = count_files_in_folder(fullpicklepath, "multipendulum_", ".pkl")
    dataset, dataset_full, masses, lengths = load_dataset_full(fullpicklepath)
    print(f"Total files: {total_files}")
    print(f"Total dataset size: {len(dataset)}")

    id_files = get_files_from_folder(os.path.join(dataset_folder, args.pickle_folder_test))
    ood_files = get_files_from_folder(os.path.join(dataset_folder, args.pickle_folder_test_outofdistr))
    id_data, _, _, _ = load_dataset_full(id_files, 0, len(id_files)-1)
    ood_data, _, _, _ = load_dataset_full(ood_files, 0, len(ood_files)-1)

    # test_pt_state = id_data[0][:1]
    # test_pt_control = id_data[]
    context = 1
    k = 5
    metric = "euclidean"
    # Find nearest neighbors
