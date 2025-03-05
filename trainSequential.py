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
    dV_dtheta = m * g * l * torch.sin(theta)
    dV_dthetadot = m * l**2 * thetadot
    ddot_theta = (-b * thetadot + m * g * l * torch.sin(theta) + u) / (m * l**2)
    dV_dt = dV_dtheta * thetadot + dV_dthetadot * ddot_theta
    return dV_dt


def lyapunov_loss(xs, ys, m=1, l=1, b=0.5, g=9.81, lambda_coeff=100.0):
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
    return lambda_coeff * lyapunov_derivative_loss.mean()


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
        loss (torch.Tensor): loss for model update.

    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    with open(file_path, 'a') as f:
        f.write(f"Iteration {i} - {args.model_logger_textfile}\n")
        f.write(f"xs\n{xs[0].detach().cpu().numpy()}\n\n")
        f.write(f"ys\n{ys[0].detach().cpu().numpy()}\n\n")
        f.write(f"output\n{output[0].detach().cpu().numpy()}\n\n")
        f.write(f"Loss ---- {loss.item()}\n\n\n\n")
        # f.write(f"lyapunov_loss_value ---- {lyapunov_loss_value.item()}\n\n\n\n")


def train_step(model, xs, ys, optimizer, loss_func, i, args):
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
    output = model(xs, ys)
    loss = loss_func(output, ys)
    # lyapunov_loss_value = lyapunov_loss(xs, output)
    total_loss = loss

    file_path = os.path.join(args.out_dir, args.model_logger_textfile)
    if i % 10 == 0:
        log_training_info(file_path, i, args, xs, ys, output, loss)

    total_loss.backward()
    grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss.detach().item(), output.detach(), grad_norm


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

def create_sliding_windows(sequence, window_size, stride):
    """
    Creates a sliding window of a given sequence with a specified window size and stride.

    Args:
        sequence (torch.Tensor): The input sequence to be windowed.
        window_size (int): The size of the window.
        stride (int): The stride of the window.

    Returns:
        torch.Tensor: A tensor containing the windowed sequences.
    """
    windows = []
    for start in range(0, len(sequence) - window_size + 1, stride):
        end = start + window_size
        windows.append(sequence[start:end])
    
    return torch.stack(windows)

def batch_create_sliding_windows(batch_sequences, window_size, stride):
    """
    Creates a sliding window of a batch of sequences with a specified window size and stride.

    Args:
        batch_sequences (torch.Tensor): The input batch of sequences to be windowed.
        window_size (int): The size of the window.
        stride (int): The stride of the window.

    Returns:
        torch.Tensor: A tensor containing the windowed sequences.
    """
    windows = []
    for sequence in batch_sequences:
        windows.append(create_sliding_windows(sequence, window_size, stride))
    
    # return torch.stack(windows, dim=0) 
    return torch.cat(windows, dim=0)

def window_dataset(xs, ys, window_size, stride):
    """
    Window the dataset into sequences of a fixed window size.

    Args:
        xs (torch.Tensor): The input sequences to be windowed.
        ys (torch.Tensor): The output sequences to be windowed.
        window_size (int): The size of the window.
        stride (int): The stride of the window.

    Returns:
        tuple: A tuple containing:
            - xs_windows (torch.Tensor): The windowed input sequences.
            - ys_windows (torch.Tensor): The windowed output sequences.
    """
    xs_windows = batch_create_sliding_windows(xs, window_size, stride)
    ys_windows = batch_create_sliding_windows(ys, window_size, stride)
    return xs_windows, ys_windows


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
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate, weight_decay=5e-4)

    curriculum = Curriculum(args.training.curriculum)
    loss_function_name = args.loss
    loss_function = getattr(tasks, loss_function_name, None)
    if loss_function is None:
        raise ValueError(f"Unknown loss function: {loss_function_name}")

    state_path = os.path.join(args.out_dir, "state.pt")

    dataset_folder = args.dataset_filesfolder
    picklefolder = args.pickle_folder
    fullpicklepath = os.path.join(dataset_folder, picklefolder)
    current_step = 0
    # current_step = 70001

    total_files = count_files_in_folder(fullpicklepath, "multipendulum_", ".pkl")
    num_chunks = args.use_chunk  
    files_per_chunk = total_files // num_chunks
    remainder = total_files % num_chunks

    # window_size = 150
    # stride = 10

    num_epochs = args.training.epochs
    # start_epoch = 28
    start_epoch = 0

    # for epoch in range(num_epochs):
    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")

        chunk_to_resume = 0
        with tqdm(total=num_chunks-chunk_to_resume, desc="Chunk Progress") as chunk_pbar: ###ebonye 150
            for chunk_idx in range(chunk_to_resume, num_chunks):
                if args.use_chunk == 1:
                    dataset = load_dataset_full(fullpicklepath)
                else:
                    start_idx = chunk_idx * files_per_chunk
                    end_idx = start_idx + files_per_chunk - 1
                    if chunk_idx == num_chunks - 1:  
                        end_idx += remainder
                    dataset = load_dataset_chunk(fullpicklepath, start_idx, end_idx)
                    

                print(f"loaded chunk {chunk_idx + 1}/{num_chunks}")

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

                #### 2/26/2025 (ebonye) batch the dataset
                my_batch_size = 16
                new_dataset = []
                x_batch = []
                y_batch = []
                for x, y in dataset:
                    x_batch.append(x)
                    y_batch.append(y)

                    if len(x_batch) == my_batch_size:
                        x_batch_merge = torch.cat(x_batch, dim=0)
                        y_batch_merge = torch.cat(y_batch, dim=0)
                        new_dataset.append((x_batch_merge, y_batch_merge))
                        x_batch = []
                        y_batch = []

                #### 2/26/2025 (ebonye) shuffle the dataset
                torch.manual_seed(epoch)
                indices = torch.randperm(len(new_dataset))
                new_dataset = [new_dataset[i] for i in indices]


                dataset = new_dataset
                del new_dataset, x_batch, y_batch
                torch.cuda.empty_cache()
                gc.collect()



                    

        

                with tqdm(total=len(dataset), desc=f"Training Chunk {chunk_idx + 1}/{num_chunks}") as pbar:
                    for xs, ys in dataset:
                        xs = xs.cuda(3)
                        ys = ys.cuda(3)


                        loss, output, gradnorm = train_step(model, xs, ys, optimizer, loss_function, current_step, args)
                        # print(f"loss: {loss}")
                        print(f"Epoch {epoch + 1}/{num_epochs}, Step {current_step}, Loss: {loss}")

                        current_step += 1
                        pbar.update(1)
                        # torch.cuda.empty_cache()
                        # gc.collect()



                        if current_step % args.wandb.log_every_steps == 0 and not args.test_run:
                            wandb.log(
                                {
                                    "epoch": epoch + 1,
                                    "step": current_step,
                                    "loss": loss,
                                    "grad_norm": gradnorm
                                }
                            )

                        curriculum.update()

                        if current_step % args.training.save_every_steps == 0 and not args.test_run:
                            training_state = {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "train_step": current_step,
                                "epoch": epoch+1,
                                "loss": loss,
                            }
                            torch.save(training_state, state_path)

                            # checkpoint_path = os.path.join(args.out_dir, f"checkpoint_{current_step}.pt")
                            checkpoint_path = os.path.join(args.out_dir, f"checkpoint_epoch{epoch+1}_step{current_step}.pt")
                            torch.save(model.state_dict(), checkpoint_path)
                            # print(f"Checkpoint saved at step {current_step}: {checkpoint_path}")
                            print(f"Checkpoint saved at epoch {epoch+1}, step {current_step}: {checkpoint_path}")
                        


                print(f"Chunk {chunk_idx + 1}/{num_chunks} finished. unloading dataset from memory...")
                del dataset
                torch.cuda.empty_cache()
                gc.collect()
                chunk_pbar.update(1)
        print(f"============== Finished Epoch {epoch + 1}/{num_epochs} ==============\n")
        

def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10
    else:
        # ##### ebonye resume run
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
    device_ids = [3, 0]
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to('cuda:3')
    # model.cuda()



    # ### ebonye
    # if args.training.resume_id is not None:
    #     checkpoint_path = os.path.join(args.out_dir, "checkpoint_epoch28_step70000.pt")
    #     print(f"checkpoint_path: {checkpoint_path}")

    #     state_path = os.path.join(args.out_dir, "state.pt")
    #     if os.path.exists(checkpoint_path):
    #         checkpoint = torch.load(checkpoint_path, map_location='cuda:3')
    #         state = torch.load(state_path, map_location='cuda:3')

           
    #         # model.load_state_dict(checkpoint['model_state_dict'])
    #         model.load_state_dict(state['model_state_dict'])
    #         optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)

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
