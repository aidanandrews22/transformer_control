import os
import random
from tqdm import tqdm
from samplers import PendulumSampler
from curriculum import Curriculum
from random import randint
import uuid
import ipdb

from quinine import QuinineArgumentParser
import torch
import yaml
from schema import schema
from models import build_model
import math
import random
import numpy as np
import torch
import pickle

# seed = [1]
torch.backends.cudnn.benchmark = True

# def reseed_all(seed):
#     """
#     Reseeds all random number generators to ensure reproducibility.

#     Args:
#         seed (int): The seed value used to initialize the random number generators.
#     """
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


def get_valid_masses_and_lengths( dt=0.01, mean=3, variance=1, lowerbound=1, upperbound=5):
    """
    Samples valid masses and lengths for a pendulum system that meet specific constraints, using a bounded Gaussian distribution.

    Args:
        dt (float, optional): The time step for simulation. Defaults to 0.01.
        mean (int, optional): The mean of the Gaussian distribution used for sampling. Defaults to 3.
        variance (int, optional): The variance of the Gaussian distribution. Defaults to 1.
        lowerbound (int, optional): The lower bound for the sampled values. Defaults to 1.
        upperbound (int, optional): The upper bound for the sampled values. Defaults to 5.

    Returns:
        tuple: A tuple containing:
            - masses (float): A valid mass value sampled from the Gaussian distribution.
            - lengths (float): A valid length value sampled from the Gaussian distribution.
    """
    while True:
        masses = sample_bounded_gaussian(mean, variance, lowerbound, upperbound)
        lengths = sample_bounded_gaussian(mean, variance, lowerbound, upperbound)
        if is_valid_mass_length(masses, lengths, dt=dt):
            return masses, lengths
        # seed[0] += 1
        # reseed_all(seed[0])

# def get_valid_masses_and_lengths_uniform( dt=0.01, masslowerbound=0.06, massupperbound=0.17, lengthlowerbound=0.2, lengthupperbound=0.55):
def get_valid_masses_and_lengths_uniform( dt=0.01, masslowerbound=0.06, massupperbound=2.06, lengthlowerbound=0.2, lengthupperbound=2.2): #### 2/24/2025 (ebonye) make wider range
    """
    Samples valid masses and lengths for a pendulum system that meet specific constraints, using a uniform distribution.

    Args:
        masslowerbound (float, optional): The lower bound for the mass value. Defaults to 0.06.
        massupperbound (float, optional): The upper bound for the mass value. Defaults to 0.17.
        lengthlowerbound (float, optional): The lower bound for the length value. Defaults to 0.2.
        lengthupperbound (float, optional): The upper bound for the length value. Defaults to 0.55.

    Returns:
        tuple: A tuple containing:
            - masses (float): A valid mass value sampled from the uniform distribution.
            - lengths (float): A valid length value sampled from the uniform distribution.
    """
    while True:
        # masses = sample_mass_uniform()
        # lengths = sample_length_uniform()
        # if is_valid_mass_length(masses, lengths, dt=0.01):
        #     return masses, lengths

        masses = sample_mass_uniform(masslowerbound, massupperbound)
        lengths = sample_length_uniform(lengthlowerbound, lengthupperbound)
        if is_valid_mass_length(masses, lengths, dt=dt):
            return masses, lengths
        # seed[0] += 1
        # reseed_all(seed[0])

def sample_bounded_gaussian(mean=3, stddev=1, lower_bound=1, upper_bound=5):
    """
    Samples a value from a Gaussian distribution within specified bounds.

    Args:
        mean (float): The mean of the Gaussian distribution.
        stddev (float): The standard deviation of the Gaussian distribution.
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.

    Returns:
        float: A sampled value within the specified bounds.
    """
    while True:
        value = random.gauss(mean, stddev)
        if lower_bound <= value <= upper_bound:
            return value
        # seed[0] += 1
        # reseed_all(seed[0])

def sample_mass_uniform(lower_bound=0.06, upper_bound=2.06):
    # before: 0.08, 0.12 #### 2/24/2025 (ebonye) make wider range
    """
    Samples a mass value from a uniform distribution within specified bounds.

    Args:
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.

    Returns:
        float: A sampled mass value within the specified bounds.
    """
    value = random.uniform(lower_bound, upper_bound)
    # seed[0] += 1
    # reseed_all(seed[0])
    return value

def sample_length_uniform(lower_bound=0.2, upper_bound=2.2):
    #before: lower_bound=0.25, upper_bound=0.45 #### 2/24/2025 (ebonye) make wider range
    """
    Samples a length value from a uniform distribution within specified bounds.

    Args:
        lower_bound (float): The lower bound of the sampled value.
        upper_bound (float): The upper bound of the sampled value.

    Returns:
        float: A sampled length value within the specified bounds.
    """
    value = random.uniform(lower_bound, upper_bound)
    # seed[0] += 1
    # reseed_all(seed[0])
    return value


def is_valid_mass_length(mass, length, dt):
    """
    Validates mass and length values to ensure they do not cause issues with rk4 during simulation.

    Args:
        mass (float): The mass of the pendulum.
        length (float): The length of the pendulum.
        dt (float): The time step for the simulation.

    Returns:
        bool: True if the mass and length values are valid, False otherwise.
    """
    g = 9.81 
    natural_frequency = math.sqrt(g / length)  
    moment_of_inertia = mass * length**2
    if moment_of_inertia < 1e-6: 
        return False
    if natural_frequency * dt > 0.1: 
        return False
    return True

def generate_random_X0_old():
    """
    Generates a random initial state for a pendulum system.

    Returns:
        list: A list containing:
            - theta (float): A randomly generated theta, sampled uniformly from range [-π, π].
            - thetadot (float): A randomly generated thetadot, sampled uniformly from the range [-10, 10].
    """
    # theta = np.random.uniform(-np.pi, np.pi)
    # thetadot = np.random.uniform(-10, 10)

    theta = np.random.uniform(-np.pi/6, np.pi/6) ######2/8/2025 (ebonye): thirty degree recommended by gpt
    thetadot = np.random.uniform(-3,3) ######2/8/2025 (ebonye): three rad/s recommended by gpt

    ###### 2/5/2025 (ebonye): same init cond for training
    # epsilon = 1e-6  
    # theta_ranges = [(-3 * np.pi / 2, -np.pi - epsilon), (np.pi + epsilon, 3 * np.pi / 2)]
    # theta_choice = np.random.choice([0, 1])
    # theta = np.random.uniform(*theta_ranges[theta_choice])
    # thetadot_ranges = [(-20.0, -11.0), (11.0, 20.0)]
    # thetadot_choice = np.random.choice([0, 1])
    # thetadot = np.random.uniform(*thetadot_ranges[thetadot_choice])
    return [theta, thetadot]

def generate_random_X0(theta_range=(-np.pi, np.pi), thetadot_range=(-3, 3)):
    """
    Generates a random initial state for a pendulum system.

    Returns:
        list: A list containing:
            - theta (float): A randomly generated theta, sampled uniformly from range [-π, π].
            - thetadot (float): A randomly generated thetadot, sampled uniformly from the range [-10, 10].
    """
    theta = np.random.uniform(*theta_range)
    thetadot = np.random.uniform(*thetadot_range)
    return [theta, thetadot]

def save_pickle(data, pickle_path):
    """
    Saves data to a pickle file.

    Args:
        data (any): The data to save.
        pickle_path (str): The path to the pickle file where the data will be saved.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f)


# def append_to_seed_file(seed_file, iteration, seed, masses, lengths, k_values, xs_shape):
#     """
#     Appends simulation details to a seed file for tracking and reproducibility.

#     Args:
#         seed_file (str): The path to the seed file.
#         iteration (int): The iteration number of the simulation.
#         seed (int): The seed value used for the simulation.
#         masses (list[float]): The mass used in the simulation.
#         lengths (list[float]): The length used in the simulation.
#         k_values (list[float]): The gain matrix used in the simulation.
#         xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
#     """
#     with open(seed_file, 'a') as f:
#         f.write(f"Iteration: {iteration}\n")
#         f.write(f"Seed: {seed}\n")
#         f.write("Masses: " + str(masses) + "\n")
#         f.write("Lengths: " + str(lengths) + "\n")
#         f.write("K: " + str(k_values) + "\n")
#         f.write(f"xs.shape: {xs_shape}\n")
#         f.write("-" * 10 + "\n")

def append_to_dataset_logger(iteration, masses, lengths, k_values, xs_shape, log_file):
    """
    Appends simulation details to a seed file for tracking and reproducibility.

    Args:
        iteration (int): The iteration number of the simulation.
        masses (list[float]): The mass used in the
        lengths (list[float]): The length used in the simulation.
        k_values (list[float]): The gain matrix used in the simulation.
        xs_shape (tuple): The shape of the state dataset (batch_size, n_dim, timepoints).
        log_file (str): The path to the log file.
    """
    with open(log_file, 'a') as f:
        f.write(f"Iteration: {iteration}\n")
        f.write("Masses: " + str(masses) + "\n")
        f.write("Lengths: " + str(lengths) + "\n")
        f.write("K: " + str(k_values) + "\n")
        f.write(f"xs.shape: {xs_shape}\n")
        f.write("-" * 10 + "\n")


def make_train_data(args):
    """
    Geneates training datasets for an inverted pendulum system simulation and saves them 
    as pickle files along with metadata for reproducibility.

    Args:
        args (Namespace):
            - args.training.train_steps (int): The total number of pickle files.
            - args.training.batch_size (int): batch of each pickle file.
            - args.training.curriculum (dict): Curriculum settings to adjust training parameters dynamically.
            - args.dataset_filesfolder (str): The directory where dataset files and logs are stored.
            - args.dataset_logger_textfile (str): The name of the file for logging dataset info.
            - args.pickle_folder (str): The dataset_filesfolder subfolder where generated dataset pickle files are saved.

    Notes:
        - The `PendulumSampler` is used to generate the dataset based on random valid pendulum parameters (masses and lengths).
        - The generated datasets are saved as pickle files named in the format `multipendulum_{i}.pkl`.
        - Metadata such as the current seed, pendulum parameters, and dataset shape is logged in a separate file for reproducibility.
        - The curriculum dynamically updates training parameters, such as the number of points in each dataset.

    """
    curriculum = Curriculum(args.training.curriculum)
    starting_step = 0
    bsize = args.training.batch_size
    pbar = tqdm(range(starting_step, args.training.train_steps + args.training.test_pendulums + args.training.test_pendulums_outofdistr)) 
    # pbar_test = tqdm(range(args.training.test_pendulums))
    # num_test_pendulums = args.training.test_pendulums

    # seed_file = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    train_logger = os.path.join(args.dataset_filesfolder, args.dataset_logger_textfile)
    test_logger = os.path.join(args.dataset_filesfolder, args.dataset_test_logger_textfile)
    test_logger_outofdistr = os.path.join(args.dataset_filesfolder, args.dataset_test_outofdistr_logger_textfile) ## 3/5/2025 out of distribution data
    base_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder)
    test_data_dir = os.path.join(args.dataset_filesfolder, args.pickle_folder_test)
    test_data_dir_outofdistr = os.path.join(args.dataset_filesfolder, args.pickle_folder_test_outofdistr) ## 3/5/2025 out of distribution data
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(test_data_dir_outofdistr, exist_ok=True) ## 3/5/2025 out of distribution data
    
    # easy_data_dir = os.path.join(base_data_dir, "easy")
    # medium_data_dir = os.path.join(base_data_dir, "medium")
    # hard_data_dir = os.path.join(base_data_dir, "hard")
    # extreme_data_dir = os.path.join(base_data_dir, "extreme")
    # os.makedirs(easy_data_dir, exist_ok=True)
    # os.makedirs(medium_data_dir, exist_ok=True)
    # os.makedirs(hard_data_dir, exist_ok=True)
    # os.makedirs(extreme_data_dir, exist_ok=True)

    # X0 = generate_random_X0()
    # X0 = [0.46671834184573213, -2.639731918107688] #### 2/24/2025 (ebonye) 4c4aa2ff... model
    # X0 = [np.pi/2, 3]

    ##### 3/6/2025 (ebonye) curriculum learning
    # easy_traj = 22000 #28333 #int(args.training.train_steps * 0.4)
    # medium_traj = 17000 #13333 #int(args.training.train_steps * 0.3)
    # hard_traj = 14000 #5834 #int(args.training.train_steps * 0.2)
    # extreme_traj = 10,000 #2500 #args.training.train_steps - easy_traj - medium_traj - hard_traj
    for i in pbar:
        # reseed_all(seed[0]+i)
        
        # masses, lengths = get_valid_masses_and_lengths_uniform()
        # # sampler = PendulumSampler(n_dims=2, init_conditions=X0)
        # sampler = PendulumSampler(n_dims=2)
        # # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, mass = masses, length = lengths)
        # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
        
        if i < args.training.train_steps:
            masses, lengths = get_valid_masses_and_lengths_uniform()
            # sampler = PendulumSampler(n_dims=2, init_conditions=X0)
            sampler = PendulumSampler(n_dims=2)
            # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, mass = masses, length = lengths)
            T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
            pickle_file = f'multipendulum_{i}.pkl'
            pickle_path = os.path.join(base_data_dir, pickle_file)
            # save_pickle((xs, control_values), pickle_path)
            save_pickle((xs, control_values, masses, lengths), pickle_path)
            append_to_dataset_logger(i, masses, lengths, k_values, xs.shape, train_logger)
        
        # if i < easy_traj:
        #     masses, lengths = get_valid_masses_and_lengths_uniform(masslowerbound=0.06, massupperbound=0.08, lengthlowerbound=0.2, lengthupperbound=0.25)
        #     X0 = generate_random_X0(theta_range=(-np.pi/12, np.pi/12), thetadot_range=(-0.5, 0.5))
        #     sampler = PendulumSampler(n_dims=2, init_conditions=X0)
        #     T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
        #     pickle_file = f'multipendulum_{i}.pkl'
        #     pickle_path = os.path.join(easy_data_dir, pickle_file)
        #     save_pickle((xs, control_values), pickle_path)
        #     append_to_dataset_logger(i, masses, lengths, k_values, xs.shape, train_logger)
        # elif i >= easy_traj and i < easy_traj + medium_traj:
        #     mass, lengths = get_valid_masses_and_lengths_uniform(masslowerbound=0.08, massupperbound=0.13, lengthlowerbound=0.25, lengthupperbound=0.45)
        #     X0 = generate_random_X0(theta_range=(-np.pi/3, np.pi/3), thetadot_range=(-1, 1))
        #     sampler = PendulumSampler(n_dims=2, init_conditions=X0)
        #     T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
        #     pickle_file = f'multipendulum_{i - easy_traj}.pkl'
        #     pickle_path = os.path.join(medium_data_dir, pickle_file)
        #     save_pickle((xs, control_values), pickle_path)
        #     append_to_dataset_logger(i, masses, lengths, k_values, xs.shape, train_logger)
        # elif i >= easy_traj + medium_traj and i < easy_traj + medium_traj + hard_traj:
        #     masses, lengths = get_valid_masses_and_lengths_uniform(masslowerbound=0.13, massupperbound=0.17, lengthlowerbound=0.45, lengthupperbound=0.55)
        #     X0 = generate_random_X0(theta_range=(-np.pi/2, np.pi/2), thetadot_range=(-2, 2))
        #     sampler = PendulumSampler(n_dims=2, init_conditions=X0)
        #     T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
        #     pickle_file = f'multipendulum_{i-easy_traj-medium_traj}.pkl'
        #     pickle_path = os.path.join(hard_data_dir, pickle_file)
        #     save_pickle((xs, control_values), pickle_path)
        #     append_to_dataset_logger(i, masses, lengths, k_values, xs.shape, train_logger)
        # elif i >= easy_traj + medium_traj + hard_traj and i < args.training.train_steps:
        #     masses, lengths = get_valid_masses_and_lengths_uniform(masslowerbound=0.13, massupperbound=0.17, lengthlowerbound=0.45, lengthupperbound=0.55)
        #     X0 = generate_random_X0(theta_range=(-np.pi, np.pi), thetadot_range=(-3, 3))
        #     sampler = PendulumSampler(n_dims=2, init_conditions=X0)
        #     T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
        #     pickle_file = f'multipendulum_{i-easy_traj-medium_traj-hard_traj}.pkl'
        #     pickle_path = os.path.join(extreme_data_dir, pickle_file)
        #     save_pickle((xs, control_values), pickle_path)
        #     append_to_dataset_logger(i, masses, lengths, k_values, xs.shape, train_logger)

        elif i >= args.training.train_steps and i < args.training.train_steps + args.training.test_pendulums:
            masses, lengths = get_valid_masses_and_lengths_uniform()
            # X0 = generate_random_X0(theta_range=(-np.pi, np.pi), thetadot_range=(-3, 3))
            # sampler = PendulumSampler(n_dims=2, init_conditions=X0)
            sampler = PendulumSampler(n_dims=2)
            # T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, bsize, mass = masses, length = lengths)
            T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
            pickle_file = f'multipendulum_test_{i-args.training.train_steps}.pkl'
            pickle_path = os.path.join(test_data_dir, pickle_file)
            # save_pickle((xs, control_values), pickle_path)
            save_pickle((xs, control_values, masses, lengths), pickle_path)
            append_to_dataset_logger(i-args.training.train_steps, masses, lengths, k_values, xs.shape, test_logger)
        else:
            # masses, lengths = get_valid_masses_and_lengths_uniform(masslowerbound=0.2, massupperbound=0.3, lengthlowerbound=0.6, lengthupperbound=0.85) ## 3/5/2025 out of distribution data
            masses, lengths = get_valid_masses_and_lengths_uniform(masslowerbound=2.07, massupperbound=3.07, lengthlowerbound=2.3, lengthupperbound=3.3) ## 3/5/2025 out of distribution data
            # sampler = PendulumSampler(n_dims=2)
            # X0 = generate_random_X0(theta_range=(-np.pi, np.pi), thetadot_range=(-3, 3))
            # sampler = PendulumSampler(n_dims=2, init_conditions=X0)
            sampler = PendulumSampler(n_dims=2)
            T, xs, control_values, k_values = sampler.generate_xs_dataset(curriculum.n_points, mass = masses, length = lengths)
            pickle_file = f'multipendulum_test_outofdistr_{i-args.training.train_steps-args.training.test_pendulums}.pkl'
            pickle_path = os.path.join(test_data_dir_outofdistr, pickle_file)
            # save_pickle((xs, control_values), pickle_path)
            save_pickle((xs, control_values, masses, lengths), pickle_path)
            append_to_dataset_logger(i-args.training.train_steps-args.training.test_pendulums, masses, lengths, k_values, xs.shape, test_logger_outofdistr)

        # append_to_seed_file(seed_file, i, seed[0] + i, masses, lengths, k_values, xs.shape)
        curriculum.update()
    
    

def main(args):
    # reseed_all(seed[0])
    global_seed = 42
    random.seed(global_seed)
    make_train_data(args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    os.makedirs(args.dataset_filesfolder, exist_ok=True)
    with open(os.path.join(args.dataset_filesfolder, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)

