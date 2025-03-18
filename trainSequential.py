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


def calculate_lyapunov_derivative(theta, thetadot, u, m=1, l=1, b=0.5, g=9.81):
    """
    Calculates the time derivative of the Lyapunov function for the inverted pendulum system,
    which is used to evaluate the stability of the system under a given control input.

    Args:
        theta (torch.Tensor): The angular position of the pendulum (in radians).
        thetadot (torch.Tensor): The angular velocity of the pendulum.
        u (torch.Tensor): The control input applied to the system.
        m (float, optional): The mass of the pendulum. Defaults to 1.
        l (float, optional): The length of the pendulum. Defaults to 1.
        b (float, optional): The damping coefficient. Defaults to 0.5.
        g (float, optional): Gravity (in m/s^2). Defaults to 9.81.

    Returns:
        torch.Tensor: The time derivative of the Lyapunov function dV/dt, showing if energy is decreasing at all time steps
    """
    ### Note that V = 0.5 * m * l^2 * thetadot^2 + m * g * l * (1 - cos(theta))
    m = m.to(theta.device)
    l = l.to(theta.device)

    dV_dtheta = m[:, None] * g * l[:, None] * torch.sin(theta)
    dV_dthetadot = m[:, None] * l[:, None]**2 * thetadot
    ddot_theta = (-b * thetadot + m[:, None] * g * l[:, None] * torch.sin(theta) + u) / (m[:, None] * l[:, None]**2)
    dV_dt = dV_dtheta * thetadot + dV_dthetadot * ddot_theta
    return dV_dt


def lyapunov_loss(xs, ys, m=1, l=1, b=0.5, g=9.81, lambda_coeff=100):
    """
    Getting the Lyapunov loss, which penalizes positive derivatives of the Lyapunov function

    Args:
        xs (torch.Tensor): The state trajectory of the system, with shape (batch_size, timesteps, 2),
            where last dimension is [theta, thetadot].
        ys (torch.Tensor): The control input of system, with shape (batch_size, u value).
        m (float, optional): The mass of the pendulum. Defaults to 1.
        l (float, optional): The length of the pendulum. Defaults to 1.
        b (float, optional): The damping coefficient. Defaults to 0.5.
        g (float, optional): gravity (in m/s^2). Defaults to 9.81.
        lambda_coeff (float, optional): Scaling loss. Defaults to 100.0.

    Returns:
        torch.Tensor: The mean Lyapunov loss, penalizes positive derivatives of the Lyapunov function.
    """
    theta = xs[:, :, 0]  
    thetadot = xs[:, :, 1]  
    u = ys  
    dV_dt = calculate_lyapunov_derivative(theta, thetadot, u, m, l, b, g)
    lyapunov_derivative_loss = torch.clamp(dV_dt, min=0)
    # lyapunov_derivative_loss = torch.exp(lambda_coeff * dV_dt) - 1
    # lyapunov_derivative_loss = torch.nn.functional.softplus(lambda_coeff * dV_dt)
    # return lyapunov_derivative_loss.mean()

    return lambda_coeff * lyapunov_derivative_loss.mean()
    # return lambda_coeff * dV_dt.mean()

def dynamics_consistency_loss(model, xs, ys, m=1, l=1, b=0.5, g=9.81, lambda_coeff=10):
    """
    Getting loss that penalizes thetadotdot values not close to true thetadotdot values

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        xs (torch.Tensor): The state trajectory of the system, with shape (batch_size, timesteps, 2),
            where last dimension is [theta, thetadot].
        ys (torch.Tensor): The control input of system, with shape (batch_size, u value).
        m (float, optional): The mass of the pendulum. Defaults to 1.
        l (float, optional): The length of the pendulum. Defaults to 1.
        b (float, optional): The damping coefficient. Defaults to 0.5.
        g (float, optional): gravity (in m/s^2). Defaults to 9.81.
        lambda_coeff (float, optional): Scaling loss. Defaults to 100.0.
    """
    theta = xs[:, :, 0]  
    thetadot = xs[:, :, 1]  
    u_true = ys
    u_pred = model(xs, ys)
    ddot_theta_true = (-b * thetadot + m[:, None] * g * l[:, None] * torch.sin(theta) + u_true) / (m[:, None] * l[:, None]**2)
    ddot_theta_pred = (-b * thetadot + m[:, None] * g * l[:, None] * torch.sin(theta) + u_pred) / (m[:, None] * l[:, None]**2)
    return lambda_coeff * torch.nn.functional.mse_loss(ddot_theta_pred, ddot_theta_true)



def rk4_step_combined(state, u, mass, length, dt=0.01, b=0.5, g=9.81):
    """
    Performs one RK4 integration step with batch support for the inverted pendulum system.

    Args:
        state (torch.Tensor): Shape (batch_size, 2), current [theta, thetadot] values.
        u (torch.Tensor): Shape (batch_size, 1), control input values.
        mass (float): Mass of the pendulum.
        length (float): Length of the pendulum.
        dt (float, optional): Time step for integration. Defaults to 0.01.
        b (float, optional): Damping coefficient. Defaults to 0.5.
        g (float, optional): Gravity. Defaults to 9.81.

    Returns:
        torch.Tensor: Shape (batch_size, 2), updated [theta, thetadot] values.
    """

    device = state.device
    dtype = state.dtype
    # mass = torch.tensor(mass, dtype=dtype, device=device)
    # length = torch.tensor(length, dtype=dtype, device=device)
    mass = mass.clone().detach().requires_grad_(True).to(device)
    length = length.clone().detach().requires_grad_(True).to(device)
    b = torch.tensor(b, dtype=dtype, device=device)
    g = torch.tensor(g, dtype=dtype, device=device)
    dt = torch.tensor(dt, dtype=dtype, device=device)
    # mass


    def pendulum_dynamics(state, u, mass, length, b, g):
        theta, thetadot = state[..., 0], state[..., 1]
        dtheta = thetadot
        dthetadot = (-b * thetadot + mass * g * length * torch.sin(theta) + u) / (mass * length**2)
        return torch.stack([dtheta, dthetadot], dim=-1) # shape (batch_size, 2)

    # Perform RK4 integration
    k1 = pendulum_dynamics(state, u, mass, length, b, g)
    k2 = pendulum_dynamics(state + 0.5 * dt * k1, u, mass, length, b, g)
    k3 = pendulum_dynamics(state + 0.5 * dt * k2, u, mass, length, b, g)
    k4 = pendulum_dynamics(state + dt * k3, u, mass, length, b, g)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)




def log_training_info(file_path, i, args, xs, ys, output, loss):
    """
    Logs training information to txt file

    Args:
        file_path (str): This will be path of where model is saved.
        i (int): The current training iteration.
        args (Namespace): schema arguments
        xs (torch.Tensor): These are theta and thetadot values
        ys (torch.Tensor): The are ground truth control u values
        output (torch.Tensor): These are predicted control u values.
        output_states (torch.Tensor): These are predicted theta and thetadot values.
        loss (torch.Tensor): loss for model update.

    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"Iteration {i} - {args.model_logger_textfile}\n")
        f.write(f"xs\n{xs[0].detach().cpu().numpy()}\n\n")
        f.write(f"ys\n{ys[0].detach().cpu().numpy()}\n\n")
        f.write(f"output\n{output[0].detach().cpu().numpy()}\n\n")
        # f.write(f"output_states\n{output_states[0]}\n\n")
        f.write(f"Loss ---- {loss.item()}\n\n\n\n")
        # f.write(f"lyapunov_loss_value ---- {lyapunov_loss_value.item()}\n\n\n\n")


# def train_step(model, xs, ys, optimizer, loss_func, i, args):
def train_step(model, xs, ys, optimizer, loss_func, i, args, mass, length, b=0.5, g=9.81):
    """
    Performs a single training step for the model, including forward pass, loss calculation, 
    backpropagation, and optimizer update. Also logs every 500 iteration.

    Args:
        model (torch.nn.Module): GPT2 decoder.
        xs (torch.Tensor): These are theta and thetadot values
        ys (torch.Tensor): These are ground truth control u values
        optimizer (torch.optim.Optimizer): Adam optimizer
        loss_func (callable): loss function that will be used for training loss.
        i (int): current training iteration.
        args (Namespace): schema arguments

    Returns:
        tuple: A tuple containing:
            - total_loss (float): total loss value for the current iteration.
            - output (torch.Tensor): The model's predicted outputs for the input data (don't really use this for anything).
    """
    optimizer.zero_grad()
    context = np.random.randint(0, (xs.size(1)//4))
    output = model(xs, ys)
    # loss = loss_func(output[:, context:], ys[:, context:])
    alpha = 0.2
    weights = 1 + alpha * torch.abs(ys[:, context:])
    loss = torch.mean(weights * (output[:, context:] - ys[:, context:])**2)
    # import pdb; pdb.set_trace()
    # loss = loss_func(output, ys)
    # lambda_coeff2 = 1e-4
    # smoothness_loss = lambda_coeff2 * torch.mean((output[:, 2:] - 2 * output[:, 1:-1] + output[:, :-2])**2) * 1e8
    # lyapunov_loss_value = lyapunov_loss(xs, output, mass, length, b, g)

    # ###### ebonye 3/17/2025
    # optimizer.zero_grad()
    # # context = np.random.randint(0, (xs.size(1)//4))
    # context = torch.randint(low = 2, high = (xs.size(1)//4), size=(1,)).item()
    # # xs_context = xs[:, :context, :]
    # ys_context = ys[:, :context]
    # # ys_true_after_context = ys[:, context:]
    # # pred_controls = torch.zeros(xs.size(0), xs.size(1)-context, 1)
    # loss = 0.0
    # # pred_controls = torch.zeros_like(ys[:, context:])
    # for j in range(context, xs.size(1)):
    #     # output = model(xs_context, ys_context)[:, -1]
    #     output = model(xs[:, :j, :], ys_context)[..., -1]
    #     # u_pred = output[:, -1]
    #     # x_pred = rk4_step_combined(xs_context[:, -1], u_pred, mass, length, dt=0.01, b=b, g=g)
    #     # x_pred = xs[:, j, :]
    #     loss += loss_func(output, ys[:, j])

    #     # xs_context = torch.cat([xs_context, x_pred.unsqueeze(1)], dim=1)
    #     ys_context = torch.cat([ys_context, output.unsqueeze(1)], dim=1)
    #     output = output.detach()
    #     del output
    #     # pred_controls[:, j-context] = u_pred

    # # loss = loss_func(pred_controls, ys_true_after_context)
    # del ys_context

    # loss = loss / (xs.size(1) - context)
    #############################


    total_loss = loss #+ smoothness_loss
    # total_loss = lyapunov_loss_value
    total_loss = total_loss.to(xs.device).requires_grad_(True)

    # import pdb; pdb.set_trace()
    # print("This function is running")

    file_path = os.path.join(args.out_dir, args.model_logger_textfile)
    if i % 10 == 0:
        log_training_info(file_path, i, args, xs, ys, output, total_loss)

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    optimizer.step()
    return total_loss.detach().item(), output.detach(), grad_norm

def train_step_with_rk4(model, xs, ys, optimizer, loss_func, i, args, mass, length, b=0.5, g=9.81, dt=0.01):
    """
    Performs a single training step for the model, including forward pass, loss calculation, 
    backpropagation, and optimizer update. Also logs every 500 iteration. Includes RK4 integration.

    Args:
        model (torch.nn.Module): GPT2 decoder.
        xs (torch.Tensor): These are theta and thetadot values for batch, shape (batch_size, timesteps, 2)
        ys (torch.Tensor): These are ground truth control u values for batch, shape (batch_size, timesteps, 1)
        optimizer (torch.optim.Optimizer): Adam optimizer
        loss_func (callable): loss function that will be used for training loss.
        i (int): current training iteration.
        args (Namespace): schema arguments
        mass (float): Mass of the pendulum.
        length (float): Length of the pendulum.
        b (float, optional): Damping coefficient. Defaults to 0.5.
        g (float, optional): Gravity. Defaults to 9.81.
        dt (float, optional): Time step for integration. Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - total_loss (float): total loss value for the current iteration.
            - output (torch.Tensor): The model's predicted outputs for the input data (don't really use this for anything).
    """
    optimizer.zero_grad()

    # Initial state for the batch (first state in the trajectory, shape (batch_size, 2))
    batch_size = xs.size(0)
    state = xs[:, 0, :]
    

    # Define the target state as (0, 0) for the pendulum
    # target_state = torch.zeros_like(state)
    target_state = torch.zeros_like(xs[:, 0, :])

    # Initialize total loss
    # total_loss = 0.0
    # total_loss = torch.tensor(0.0, dtype=torch.float32, device=xs.device, requires_grad=True)
    # total_loss = total_loss.requires_grad_(True)
    total_loss = torch.zeros(1, device=xs.device)
    # lyapunov = lyapunov_loss(xs, ys, mass, length, b, g)

    # List to keep track of states and actions for logging
    all_states = []
    all_actions = []

    # get all controls from the model
    controls = model(xs, ys)

    state.requires_grad_(True)

    # Loop over timesteps in the trajectory
    for t in range(xs.size(1) - 1):
        # Get control input for current timestep
        # u = ys[:, t, :]

        # Get predicted control input for current timestep from the model based on current state   
        # u = model(state)
        u = controls[:, t]

        # Record states and actions for logging
        all_states.append(state.detach().cpu().numpy())
        all_actions.append(u.detach().cpu().numpy())

        # Perform RK4 integration step
        # state = rk4_step_combined(state, u, mass, length, dt, b, g)
        state = rk4_step_combined(xs[:, t, :], u, mass, length, dt, b, g)

        # Calculate loss for the current timestep from target state
        # print(f"Predicted state: {state}")  
        # print(f"Target state: {target_state}")
        # loss = loss_func(state, target_state) ### minimize the distance between the predicted state and target state

        loss = loss_func(state, xs[:, t+1, :]) ### minimize the distance between the predicted state and the next state
        # total_loss += loss
        total_loss = total_loss + loss
        # import pdb; pdb.set_trace()
        # print(f"Loss: {loss}")

    file_path = os.path.join(args.out_dir, args.model_logger_textfile)
    # if i % 10 == 0:
    #     # log_training_info(file_path, i, args, xs, ys, output, loss)
    #     log_training_info(file_path, i, args, xs, ys, all_actions, all_states, total_loss)

    total_loss = total_loss / (xs.size(1)-1)
    # total_loss = total_loss + lyapunov

    total_loss.backward()
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(f"Parameter: {name}, Gradient norm: {param.grad.norm().item()}")
    #     else:
    #         print(f"Parameter: {name}, Gradient is None")
    
    grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    
    optimizer.step()
    # return total_loss.detach().item(), output.detach(), grad_norm

    return total_loss.detach().item(), all_actions, all_states, grad_norm


    




    

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
    with tqdm(total=end_idx - start_idx + 1, desc=f"Loading files {start_idx}-{end_idx}", leave=False) as load_pbar:
        for i in range(start_idx, end_idx + 1):
            pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    xs, ys = pickle.load(f)
                    dataset.append((xs, ys))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset

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
    total_files = count_files_in_folder(pickle_folder, "multipendulum_", ".pkl")
    with tqdm(total=total_files, desc="Loading all files", leave=False) as load_pbar:
        for i in range(total_files):
            pickle_path = os.path.join(pickle_folder, f"multipendulum_{i}.pkl")
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    xs, ys = pickle.load(f)
                    dataset.append((xs, ys))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset

def load_dataset_full_with_rk4(pickle_folder):
    """
    Loads entire dataset from a specified folder

    Args:
        pickle_folder (str): The path to the folder containing the pickle files.

    Returns:
        list: A list of tuples, where each tuple contains:
            - xs (any): theta and thetadot values.
            - ys (any): control u values.
        list: A list of mass values for each trajectory.
        list: A list of length values for each trajectory.
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
                    xs, ys, mass, length = pickle.load(f)
                    dataset.append((xs, ys))
                    masses.append(mass)
                    lengths.append(length)
                    dataset_full.append((xs, ys, mass, length))
            else:
                print(f"Pickle not found: {pickle_path}. Skipping...")
            load_pbar.update(1)
    return dataset, dataset_full, masses, lengths

def collate_fn(batch):
    """
    Collate function to handle the batching of the dataset, ensuring that mass/length 
    and corresponding states/controls are grouped together properly.
    
    Args:
        batch (list): A list of tuples, where each tuple contains:
            - xs (torch.Tensor): theta and thetadot values.
            - ys (torch.Tensor): control u values.
            - mass (float): mass of the pendulum.
            - length (float): length of the pendulum.

    Returns:
        tuple: A tuple containing:
            - xs (torch.Tensor): theta and thetadot values for the batch.
            - ys (torch.Tensor): control u values for the batch.
            - masses (torch.Tensor): mass values for the batch.
            - lengths (torch.Tensor): length values for the batch.
    """
    # xs, ys, masses, lengths = zip(*batch)
    # xs = torch.tensor(xs, dtype=torch.float32)
    # ys = torch.tensor(ys, dtype=torch.float32)
    # masses = torch.tensor(masses, dtype=torch.float32)
    # lengths = torch.tensor(lengths, dtype=torch.float32)

    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    masses = [item[2] for item in batch]
    lengths = [item[3] for item in batch]

    xs = torch.squeeze(torch.stack(xs, dim=0))
    ys = torch.squeeze(torch.stack(ys, dim=0))

    masses = torch.tensor(masses, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.float32)

    return xs, ys, masses, lengths

##############################################################################################################
###### ebonye 3/6/2025
def sample_trajectory_files(phase, trajectory_files, num_trajectories):
    if phase ==1:
        return np.random.choice(trajectory_files["easy"], num_trajectories, replace=False)
    elif phase == 2:
        easy_samples = np.random.choice(trajectory_files["easy"], num_trajectories//2, replace=False)
        medium_samples = np.random.choice(trajectory_files["medium"], num_trajectories//2, replace=False)
        return list(easy_samples) + list(medium_samples)
    elif phase == 3:
        easy_samples = np.random.choice(trajectory_files["easy"], num_trajectories//3, replace=False)
        medium_samples = np.random.choice(trajectory_files["medium"], num_trajectories//3, replace=False)
        hard_samples = np.random.choice(trajectory_files["hard"], num_trajectories//3, replace=False)
        return list(easy_samples) + list(medium_samples) + list(hard_samples)
    elif phase == 4:
        easy_samples = np.random.choice(trajectory_files["easy"], num_trajectories//4, replace=False)
        medium_samples = np.random.choice(trajectory_files["medium"], num_trajectories//4, replace=False)
        hard_samples = np.random.choice(trajectory_files["hard"], num_trajectories//4, replace=False)
        extreme_samples = np.random.choice(trajectory_files["extreme"], num_trajectories//4, replace=False)
        return list(easy_samples) + list(medium_samples) + list(hard_samples) + list(extreme_samples)
    else:
        return ValueError("Invalid phase number")
    

def get_files_from_folder(folderpath):
    """ Get all .pkl files from a folder """
    return[os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith(".pkl")]

def get_files_for_phase(phase, easy_files, medium_files, hard_files, extreme_files, used_files):
    """Get the list of files for the current phase"""
    if phase == 1:
        available = [f for f in easy_files if f not in used_files["easy"]]
        selected = list(np.random.choice(available, 15000, replace=False))
        used_files["easy"].update(selected)

        available_medium = [f for f in medium_files if f not in used_files["medium"]]
        selected_medium = list(np.random.choice(available_medium, 1000, replace=False))
        used_files["medium"].update(selected_medium)
        return selected + selected_medium
    elif phase == 2:
        available_easy = [f for f in easy_files if f not in used_files["easy"]]
        selected_easy = list(np.random.choice(available_easy, 5000, replace=False))
        used_files["easy"].update(selected_easy)

        available_medium = [f for f in medium_files if f not in used_files["medium"]]
        selected_medium = list(np.random.choice(available_medium, 10000, replace=False))
        used_files["medium"].update(selected_medium)

        available_hard = [f for f in hard_files if f not in used_files["hard"]]
        selected_hard = list(np.random.choice(available_hard, 1000, replace=False))
        used_files["hard"].update(selected_hard)
        return selected_easy + selected_medium + selected_hard
    elif phase == 3:
        available_easy = [f for f in easy_files if f not in used_files["easy"]]
        selected_easy = list(np.random.choice(available_easy, 2000, replace=False)) #3333
        used_files["easy"].update(selected_easy)

        available_medium = [f for f in medium_files if f not in used_files["medium"]]
        selected_medium = list(np.random.choice(available_medium, 4000, replace=False)) #3333
        used_files["medium"].update(selected_medium)

        available_hard = [f for f in hard_files if f not in used_files["hard"]]
        selected_hard = list(np.random.choice(available_hard, 9000, replace=False)) #3334
        used_files["hard"].update(selected_hard)

        available_extreme = [f for f in extreme_files if f not in used_files["extreme"]]
        selected_extreme = list(np.random.choice(available_extreme, 1000, replace=False)) #0
        used_files["extreme"].update(selected_extreme)
        return selected_easy + selected_medium + selected_hard + selected_extreme
    elif phase == 4:
        # available_easy = [f for f in easy_files if f not in used_files["easy"]]
        # selected_easy = list(np.random.choice(available_easy, 2500, replace=False))
        # used_files["easy"].update(selected_easy)

        available_medium = [f for f in medium_files if f not in used_files["medium"]]
        selected_medium = list(np.random.choice(available_medium, 2000, replace=False)) #2500
        used_files["medium"].update(selected_medium)

        available_hard = [f for f in hard_files if f not in used_files["hard"]]
        selected_hard = list(np.random.choice(available_hard, 4000, replace=False)) #2500
        used_files["hard"].update(selected_hard)

        available_extreme = [f for f in extreme_files if f not in used_files["extreme"]]
        selected_extreme = list(np.random.choice(available_extreme, 9000, replace=False)) #2500
        used_files["extreme"].update(selected_extreme)
        return selected_medium + selected_hard + selected_extreme
    else:
        return ValueError("Invalid phase number")
    

def load_dataset_chunk_curriculum(file_list, start_idx, end_idx):
    """ Load a chunk of trajectories from the file list"""
    chunk_files = file_list[start_idx:end_idx+1]
    dataset = []
    for file in chunk_files:
        with open(file, "rb") as f:
            # xs, ys = pickle.load(f)
            xs, ys, mass, length = pickle.load(f)
            # dataset.append((xs, ys))
            dataset.append((xs, ys, mass, length))
    return dataset

def evaluate_model(model, id_data, ood_data, loss_func):
    """
    Evaluates the model on the in-distribution and out-of-distribution data.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        id_data (list): The in-distribution data to evaluate the model on.
        ood_data (list): The out-of-distribution data to evaluate the model on.
        loss_func (callable): The loss function to use for evaluation.

    Returns:
        tuple: A tuple containing:
            - id_loss (float): The loss of the model on the in-distribution data.
            - ood_loss (float): The loss of the model on the out-of-distribution data.
    """
    model.eval()
    # with torch.no_grad():
    #     id_losses = []
    #     for xs, ys in id_data:
    #         output = model(xs, ys)
    #         loss = loss_func(output, ys)
    #         id_losses.append(loss.item())
    #     id_loss = np.mean(id_losses)

    #     ood_losses = []
    #     for xs, ys in ood_data:
    #         output = model(xs, ys)
    #         loss = loss_func(output, ys)
    #         ood_losses.append(loss.item())
    #     ood_loss = np.mean(ood_losses)
    # model.train()

    id_loss = 0.0
    ood_loss = 0.0
    lambda_coeff2 = 1e-4

    id_loader = DataLoader(id_data, batch_size=64, shuffle=True)
    # for xs, ys in id_loader:
    for xs, ys, masses, lengths in id_loader:
        with torch.no_grad():
            # print(f"xs: {xs.size()}")
            # print(f"ys: {ys.size()}")
            xs = torch.squeeze(xs)
            ys = torch.squeeze(ys)
            # print(f"xs: {xs.size()}")
            # print(f"ys: {ys.size()}")
            xs = xs.cuda(3)
            ys = ys.cuda(3)
            context = torch.randint(low = 2, high = (xs.size(1)//4), size=(1,)).item()

            output = model(xs, ys)

            loss = loss_func(output[:, context:], ys[:, context:])
            # loss = loss_func(output, ys)
            # # smoothness_loss = lambda_coeff2 * torch.mean((output[:, 2:] - 2 * output[:, 1:-1] + output[:, :-2])**2) * 1e8
            # # loss = loss + smoothness_loss
        
            # loss = lyapunov_loss(xs, ys, masses, lengths)
            id_loss += loss.item()
    id_loss /= len(id_loader)

    ood_loader = DataLoader(ood_data, batch_size=64, shuffle=True)
    # for xs, ys in ood_loader:
    for xs, ys, masses, lengths in ood_loader:
        with torch.no_grad():
            xs = torch.squeeze(xs)
            ys = torch.squeeze(ys)
            xs = xs.cuda(3)
            ys = ys.cuda(3)
            context = torch.randint(low = 2, high = (xs.size(1)//4), size=(1,)).item()
            output = model(xs, ys)
            loss = loss_func(output[:, context:], ys[:, context:])
            # loss = loss_func(output, ys)
            # # smoothness_loss = lambda_coeff2 * torch.mean((output[:, 2:] - 2 * output[:, 1:-1] + output[:, :-2])**2) * 1e8
            # # loss = loss + smoothness_loss
            # loss = lyapunov_loss(xs, ys, masses, lengths)
            ood_loss += loss.item()
    ood_loss /= len(ood_loader)
    model.train()
    return id_loss, ood_loss


def evaluate_model_with_rk4(model, id_data, ood_data, loss_func, b=0.5, g=9.81, dt=0.01):
    """
    Evaluates the model on the in-distribution and out-of-distribution data.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        id_data (list): The in-distribution data to evaluate the model on.
        ood_data (list): The out-of-distribution data to evaluate the model on.
        loss_func (callable): The loss function to use for evaluation.
        b (float, optional): Damping coefficient. Defaults to 0.5.
        g (float, optional): Gravity. Defaults to 9.81.
        dt (float, optional): Time step for integration. Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - id_loss (float): The loss of the model on the in-distribution data.
            - ood_loss (float): The loss of the model on the out-of-distribution data.
    """
    model.eval()
    # id_loss = 0.0
    # ood_loss = 0.0
    id_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda:3')
    ood_loss = torch.tensor(0.0, dtype=torch.float32, device='cuda:3')

    id_loader = DataLoader(id_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for xs, ys, masses, lengths in id_loader:
        with torch.no_grad():
            xs = xs.cuda(3)
            ys = ys.cuda(3)
            masses = masses.cuda(3)
            lengths = lengths.cuda(3)

            control = model(xs, ys)

            state = xs[:, 0, :]
            total_loss = 0.0
            for t in range(xs.size(1) - 1):
                u = control[:, t]
                state = rk4_step_combined(state, u, masses, lengths, dt, b, g)
                loss = loss_func(state, xs[:, t+1, :])
                total_loss += loss.item()

            id_loss += total_loss / (xs.size(1)-1)
    id_loss /= len(id_loader)


    ood_loader = DataLoader(ood_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for xs, ys, masses, lengths in ood_loader:
        with torch.no_grad():
            xs = xs.cuda(3)
            ys = ys.cuda(3)
            masses = masses.cuda(3)
            lengths = lengths.cuda(3)
            
            control = model(xs, ys)

            state = xs[:, 0, :]
            total_loss = 0.0
            for t in range(xs.size(1) - 1):
                u = control[:, t]
                state = rk4_step_combined(state, u, masses, lengths, dt, b, g)
                loss = loss_func(state, xs[:, t+1, :])
                total_loss += loss.item()

            ood_loss += total_loss / (xs.size(1)-1)
    ood_loss /= len(ood_loader)

    model.train()
    return id_loss, ood_loss

def train(model, args):
    """
    Trains a given model on a dataset using the specified arguments and configurations.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        args (Namespace): A configuration object containing training parameters and settings, including:
            - args.training.learning_rate (float): The learning rate for the optimizer.
            - args.loss (str): The name of the loss function to use (must be defined in the tasks module).
            - args.out_dir (str): Directory where training states and checkpoints will be saved.
            - args.dataset_filesfolder (str): Path to the folder containing dataset-related files.
            - args.pickle_folder (str): dataset_filesfolder subfolder name containing the pickled dataset files.
            - args.use_chunk (int): Number of chunks to divide the dataset for memory-efficient loading. default 1
            - args.wandb.log_every_steps (int): How often to log metrics
            - args.training.save_every_steps (int): How often to save model checkpoints
            - args.test_run (bool): If True, skips logging and checkpoint saving for debugging

    Raises:
        ValueError: If the specified loss function is not found in the `tasks` module.

    Notes:
        - The function supports chunk-based dataset loading if memory restraints, or loading full dataset into memory
        - Checkpoints and training states are saved here.
        - Wandb logs are done here.
    """
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate, weight_decay=1e-4) #not good
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=5e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=7e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=1e-2)

    curriculum = Curriculum(args.training.curriculum)
    loss_function_name = args.loss
    loss_function = getattr(tasks, loss_function_name, None)
    if loss_function is None:
        raise ValueError(f"Unknown loss function: {loss_function_name}")

    state_path = os.path.join(args.out_dir, "state.pt")

    dataset_folder = args.dataset_filesfolder
    picklefolder = args.pickle_folder
    fullpicklepath = os.path.join(dataset_folder, picklefolder)

    #### indistribution and out of distribution data
    id_files = get_files_from_folder(os.path.join(dataset_folder, args.pickle_folder_test))
    ood_files = get_files_from_folder(os.path.join(dataset_folder, args.pickle_folder_test_outofdistr))
    id_data = load_dataset_chunk_curriculum(id_files, 0, len(id_files)-1)
    ood_data = load_dataset_chunk_curriculum(ood_files, 0, len(ood_files)-1)
    # _, id_data, _, _ = load_dataset_full_with_rk4(os.path.join(dataset_folder, args.pickle_folder_test))
    # _, ood_data, _, _ = load_dataset_full_with_rk4(os.path.join(dataset_folder, args.pickle_folder_test_outofdistr))

    #### only random 256 in-distribution and out-of-distribution data
    indices = torch.randperm(len(id_data))
    id_data = [id_data[i] for i in indices[:256]]
    indices = torch.randperm(len(ood_data))
    ood_data = [ood_data[i] for i in indices[:256]]



    ###### ebonye curriculum learning 3/6/2025
    # easy_files = get_files_from_folder(os.path.join(fullpicklepath, "easy"))
    # medium_files = get_files_from_folder(os.path.join(fullpicklepath, "medium"))
    # hard_files = get_files_from_folder(os.path.join(fullpicklepath, "hard"))
    # extreme_files = get_files_from_folder(os.path.join(fullpicklepath, "extreme"))
    # used_files = {"easy": set(), "medium": set(), "hard": set(), "extreme": set()}

    # epochs_per_phase = {
    #     1: 3,
    #     2: 12,
    #     3: 30,
    #     4: 80
    # }

    #####
    current_step = 0
    # current_step = 125001

    total_files = count_files_in_folder(fullpicklepath, "multipendulum_", ".pkl")
    num_chunks = args.use_chunk  
    files_per_chunk = total_files // num_chunks
    remainder = total_files % num_chunks

    # overall_epochs = 0
    # total_epochs = sum(epochs_per_phase.values())
    # for phase in range(1, 5):
    #     print(f"Training phase {phase}")

    num_epochs = args.training.epochs
    start_epoch = 0

    # ##### Get the files for the current phase
    # phase_files = get_files_for_phase(phase, easy_files, medium_files, hard_files, extreme_files, used_files)

    # # Shuffle the files
    # random.shuffle(phase_files)

    ###### Chunk the files
    # total_files = len(phase_files)
    # num_chunks = args.use_chunk
    # files_per_chunk = total_files // num_chunks
    # remainder = total_files % num_chunks

    chunk_to_resume = 0
    with tqdm(total=num_chunks-chunk_to_resume, desc="Chunk Progress") as chunk_pbar: ###ebonye 150
        for chunk_idx in range(chunk_to_resume, num_chunks):
            if args.use_chunk == 1:
                # dataset = load_dataset_full(fullpicklepath)
                dataset, dataset_full, masses, lengths = load_dataset_full_with_rk4(fullpicklepath)
            else:
                start_idx = chunk_idx * files_per_chunk
                end_idx = start_idx + files_per_chunk - 1
                if chunk_idx == num_chunks - 1:  
                    end_idx += remainder
                dataset = load_dataset_chunk(fullpicklepath, start_idx, end_idx)

            print(f"loaded chunk {chunk_idx + 1}/{num_chunks}")


        batch_size = 64
        dataloader = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        #### ebonye 3/15/2025 Cosine Scheduler
        num_training_steps = num_epochs * len(dataloader)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_training_steps, eta_min=1e-6)
        warmup_steps =int(0.1 * num_training_steps)
        lr_scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    # for epoch in range(num_epochs):
    for epoch in range(start_epoch, num_epochs):
    # for epoch in range(epochs_per_phase[phase]):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        # print(f"Starting epoch {epoch + 1}/{epochs_per_phase[phase]} of Phase {phase}")

        # np.random.shuffle(phase_files)

        ################ ebonye 3/15/2025 moving outside of epoch loop
        # chunk_to_resume = 0
        # with tqdm(total=num_chunks-chunk_to_resume, desc="Chunk Progress") as chunk_pbar: ###ebonye 150
        #     for chunk_idx in range(chunk_to_resume, num_chunks):
        #         if args.use_chunk == 1:
        #             # dataset = load_dataset_full(fullpicklepath)
        #             dataset, dataset_full, masses, lengths = load_dataset_full_with_rk4(fullpicklepath)
        #         else:
        #             start_idx = chunk_idx * files_per_chunk
        #             end_idx = start_idx + files_per_chunk - 1
        #             if chunk_idx == num_chunks - 1:  
        #                 end_idx += remainder
        #             dataset = load_dataset_chunk(fullpicklepath, start_idx, end_idx)
                    

        #         print(f"loaded chunk {chunk_idx + 1}/{num_chunks}")
        ################

                ### ebonye 3/6/2025 curriculum learning
                # if args.use_chunk == 1:
                #     dataset = load_dataset_chunk_curriculum(phase_files, 0, len(phase_files)-1)
                # else:
                #     start_idx = chunk_idx * files_per_chunk
                #     end_idx = start_idx + files_per_chunk - 1
                #     if chunk_idx == num_chunks - 1:  
                #         end_idx += remainder

                #     dataset = load_dataset_chunk_curriculum(phase_files, start_idx, end_idx)
                
                # print(f"loaded chunk {chunk_idx + 1}/{num_chunks} of Phase {phase} and Epoch {epoch + 1}")
                #### 2/25/2025 (ebonye) creating batches for same init cond training data
                # my_batch_size = 8 #10
                # new_dataset = []
                
                # x_batch = []
                # y_batch = []
                # for x, y in dataset:
                #     x_batch.append(x.squeeze())
                #     y_batch.append(y.squeeze())
                #     if len(x_batch) == my_batch_size:
                #         new_dataset.append((torch.stack(x_batch), torch.stack(y_batch)))
                #         x_batch = []
                #         y_batch = []
                # dataset = new_dataset

                # #### 2/25/2025 (ebonye) reformat dataset so that many mass/length in one batch
                # my_batch_size = 8
                # new_dataset = []
                # x_batch = []
                # y_batch = []
                # for x, y in dataset:
                #     x_batch.append(x)
                #     y_batch.append(y)

                # x_batch_merge = torch.cat(x_batch, dim=0)
                # y_batch_merge = torch.cat(y_batch, dim=0)

                # indices = torch.randperm(x_batch_merge.size(0))

                # x_batch_merge = x_batch_merge[indices]
                # y_batch_merge = y_batch_merge[indices]

                # for i in range(0, x_batch_merge.size(0), my_batch_size):
                #     new_dataset.append((x_batch_merge[i:i+my_batch_size], y_batch_merge[i:i+my_batch_size]))

                # dataset = new_dataset
                ##############################################
                # #### 2/26/2025 (ebonye) batch the dataset
                # # torch.manual_seed(epoch)
                # # my_batch_size = 32
                # my_batch_size = 64
                # # bigger_indicies = torch.randperm(len(dataset))
                # # dataset = [dataset[i] for i in bigger_indicies]

                # ## permute dataset before batching
                # indices_big_permute = torch.randperm(len(dataset))
                # dataset = [dataset[i] for i in indices_big_permute]

                # new_dataset = []
                # x_batch = []
                # y_batch = []
                # for x, y in dataset:
                #     x_batch.append(x)
                #     y_batch.append(y)

                #     if len(x_batch) == my_batch_size:
                #         x_batch_merge = torch.cat(x_batch, dim=0)
                #         y_batch_merge = torch.cat(y_batch, dim=0)
                #         new_dataset.append((x_batch_merge, y_batch_merge))
                #         x_batch = []
                #         y_batch = []
                    
                # if len(x_batch) > 0:
                #     x_batch_merge = torch.cat(x_batch, dim=0)
                #     y_batch_merge = torch.cat(y_batch, dim=0)
                #     new_dataset.append((x_batch_merge, y_batch_merge))

                # #### 2/26/2025 (ebonye) shuffle the dataset
                # # torch.manual_seed(epoch)
                # indices = torch.randperm(len(new_dataset))
                # new_dataset = [new_dataset[i] for i in indices]

                # dataset = new_dataset
                # del new_dataset, x_batch, y_batch
                # torch.cuda.empty_cache()
                # gc.collect()
                ##############################################
                #### 3/15/2025 (ebonye) moved to outside of epoch loop
                # #### 3/11/2025 (ebonye) dataloader with different loss fn
                # batch_size = 64
                # dataloader = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                

        

        # with tqdm(total=len(dataset), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
        with tqdm(total=len(dataloader), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
            # for xs, ys in dataset:
            for xs, ys, masses, lengths in dataloader:
                # print(f"xs: {xs}")
                # print(f"ys: {ys}")
                xs = xs.cuda(3)
                ys = ys.cuda(3)

                # print(f"xs: {xs.size()}")
                # print(f"ys: {ys.size()}")

                # import pdb; pdb.set_trace()


                # loss, output, gradnorm = train_step(model, xs, ys, optimizer, loss_function, current_step, args)
                # print(f"initial lr: {optimizer.param_groups[0]['lr']}")
                loss, output, gradnorm = train_step(model, xs, ys, optimizer, loss_function, current_step, args, masses, lengths)
                lr_scheduler.step()
                # loss, output_actions, output_states, gradnorm = train_step_with_rk4(model, xs, ys, optimizer, loss_function, current_step, args, masses, lengths)
                # import pdb; pdb.set_trace()
                # print(f"loss: {loss}")
                # print(f"Epoch {epoch + 1}/{num_epochs}, Step {current_step}, Loss: {loss}, Current LR: {lr_scheduler.get_lr()[0]}")
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {current_step}, Loss: {loss}, Current LR: {optimizer.param_groups[0]['lr']}")
                # import pdb; pdb.set_trace()

                # print(f"Phase {phase}, Epoch {epoch + 1}/{epochs_per_phase[phase]}, Step {current_step}, Loss: {loss}")
                current_step += 1
                pbar.update(1)
                # torch.cuda.empty_cache()
                # gc.collect()



                if current_step % args.wandb.log_every_steps == 0 and not args.test_run:
                    avg_id_loss, avg_ood_loss = evaluate_model(model, id_data, ood_data, loss_function)
                    # avg_id_loss, avg_ood_loss = evaluate_model_with_rk4(model, id_data, ood_data, loss_function)
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            # "epoch": overall_epochs,
                            # "phase": phase,
                            "step": current_step,
                            "loss": loss,
                            "grad_norm": gradnorm,
                            "id_loss": avg_id_loss,
                            "ood_loss": avg_ood_loss
                        }
                    )

                curriculum.update()

                if current_step % args.training.save_every_steps == 0 and not args.test_run:
                    training_state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_step": current_step,
                        "epoch": epoch+1,
                        # "epoch": overall_epochs,
                        "loss": loss,
                    }
                    torch.save(training_state, state_path)

                    # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{current_step}.pt")
                    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
                    # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{overall_epochs}_step{current_step}.pt")
                    # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_phase{phase}_epoch{epoch+1}_step{current_step}.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    # print(f"Checkpoint saved at step {current_step}: {checkpoint_path}")
                    print(f"Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
                    # print(f"Checkpoint saved at epoch {overall_epochs}, step {current_step}: {checkpoint_path}")

        print(f"Chunk {chunk_idx + 1}/{num_chunks} finished. unloading dataset from memory...")
        # del dataset
        # torch.cuda.empty_cache()
        # gc.collect()
        chunk_pbar.update(1)
        print(f"============== Finished Epoch {epoch + 1}/{num_epochs} ==============\n")
        # print(f"============== Finished Epoch {epoch + 1}/{epochs_per_phase[phase]} of Phase {phase} ==============\n")
        # overall_epochs += 1
        # print(f"Overall Epochs: {overall_epochs}/{total_epochs}")

    ##### Final Checkpoint
    training_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_step": current_step,
        "epoch": epoch+1,
        # "epoch": overall_epochs,
        "loss": loss,
    }
    torch.save(training_state, state_path)
    checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
    # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{overall_epochs}_step{current_step}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    # print(f"Final Checkpoint saved at epoch {overall_epochs}, step {current_step}: {checkpoint_path}")
    print(f"Final Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
    # print(f"============== Finished Epoch {epoch + 1}/{epochs_per_phase[phase]} of Phase {phase} ==============\n")
    # print(f"============== Finished Phase {phase} ==============\n")
                

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10
    else:
        ##### ebonye resume run
        # if os.path.exists(os.path.join(args.out_dir, "wandb", "wandb-resume.json")):
        #     with open(os.path.join(args.out_dir, "wandb", "wandb-resume.json"), "r") as f:
        #         resume_info = json.load(f)
        #         run_id = resume_info.get("run_id", None)
        #         args.training.resume_id = run_id #### ebonye
        # else:
        #     run_id = None

        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
            # id=run_id if run_id is not None else None #### ebonye
        )

    model = build_model(args.model)
    device_ids = [3, 2]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to('cuda:3')
    # model.cuda()



    ### ebonye
    # if args.training.resume_id is not None:
    #     checkpoint_path = os.path.join(args.out_dir, "checkpoint_epoch50_step125000.pt")
    #     print(f"checkpoint_path: {checkpoint_path}")

    #     state_path = os.path.join(args.out_dir, "state.pt")
    #     if os.path.exists(checkpoint_path):
    #         checkpoint = torch.load(checkpoint_path, map_location='cuda:3')
    #         state = torch.load(state_path, map_location='cuda:3')

           
    #         # model.load_state_dict(checkpoint['model_state_dict'])
    #         model.load_state_dict(state['model_state_dict'])
    #         # optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    #         optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=5e-4)


    #         # if 'optimizer_state_dict' in checkpoint:
    #         if 'optimizer_state_dict' in state:
    #             optimizer.load_state_dict(state['optimizer_state_dict'])

    #         start_step = state.get('train_step', 0) + 1
    #         loss = state.get('loss', 0.0)

    #         print(f"Resuming training from step {start_step} with loss {loss}")
    #     else:
    #         start_step = 0
    #         loss = 0.0
    #         print("Starting training from scratch 1")

    # else:
    #     start_step = 0
    #     loss = 0.0
    #     print("Starting training from scratch 2")

    
    model.train()

    train(model, args)

if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    # args.training.resume_id = "ef51e61f-9aa6-4d83-92c8-7d8681eff369"
    # args.training.resume_id = "c169175d-b2ed-4359-add7-041812fc1ab0"
    # args.training.resume_id = "9bbb6dd7-4ca0-48f5-85fa-e246a773414d"
    # args.training.resume_id = "e74aa25f-1ebe-4244-b9f2-edc7a91e2226"

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(args.out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)


    main(args)
