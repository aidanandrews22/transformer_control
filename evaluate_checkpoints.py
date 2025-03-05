import numpy as np
import torch
import matplotlib.pyplot as plt
import workCon
import os
from eval import get_model_from_run
from tqdm import tqdm
import ipdb
import traceback
import random
import math
import generate_dataset
from scipy.integrate import solve_ivp
import pickle
import re


def generate_random_X0():
    """
    Generates a random initial state for a pendulum system.

    Returns:
        list: A list containing:
            - theta (float): A randomly generated theta, sampled uniformly from range [-π, π].
            - thetadot (float): A randomly generated thetadot, sampled uniformly from the range [-10, 10].
    """
    # theta = np.random.uniform(-np.pi, np.pi)
    # thetadot = np.random.uniform(-10, 10)

    # theta = np.random.uniform(-np.pi/6, np.pi/6) ######2/8/2025 (ebonye): thirty degree recommended by gpt
    # thetadot = np.random.uniform(-3,3) ######2/8/2025 (ebonye): three rad/s recommended by gpt
    
    # theta = np.random.uniform(-np.pi/4, np.pi/4) 
    theta = np.random.uniform(np.pi/5, np.pi/2)
    thetadot = np.random.uniform(-3,3)

    ###### 2/5/2025 (ebonye): same init cond for training
    # epsilon = 1e-6  
    # theta_ranges = [(-3 * np.pi / 2, -np.pi - epsilon), (np.pi + epsilon, 3 * np.pi / 2)]
    # theta_choice = np.random.choice([0, 1])
    # theta = np.random.uniform(*theta_ranges[theta_choice])
    # thetadot_ranges = [(-20.0, -11.0), (11.0, 20.0)]
    # thetadot_choice = np.random.choice([0, 1])
    # thetadot = np.random.uniform(*thetadot_ranges[thetadot_choice])
    


    return [theta, thetadot]

def mse(theta_model, thetadot_model, theta_rk4, thetadot_rk4):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and true states.

    Args:
        theta_model (np.ndarray or list): Model's predicted theta 
        thetadot_model (np.ndarray or list): Model's predicted thetadot 
        theta_rk4 (np.ndarray or list): True theta using RK4
        thetadot_rk4 (np.ndarray or list): True thetadot using RK4

    Returns:
        float: The computed MSE value, representing the average squared difference between 
        the predicted and true states.
    """
    xs_pred = torch.tensor(np.column_stack((theta_model, thetadot_model)), dtype=torch.float32)
    xs_true = torch.tensor(np.column_stack((theta_rk4, thetadot_rk4)), dtype=torch.float32)
    return (xs_true - xs_pred).pow(2).mean().item()

def get_mass_length_Ks_from_text_file(file_path, target_iteration):
    """
    Reads the mass and length values from a text file.

    Args:
        file_path (str): The path to the text file containing the mass and length values.

    Returns:
        tuple: A tuple containing:
            - mass (float): The mass of the pendulum.
            - length (float): The length of the pendulum.
    """
    iteration_found = False
    iteration_data = {}
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            # check for iteration
            iteration_match = re.search(r'Iteration:\s*(\d+)', line)
            if iteration_match:
                current_iteration = int(iteration_match.group(1))
                if current_iteration == target_iteration:
                    iteration_found = True
                else:
                    iteration_found = False
            
            # once target iteration is found, extract mass, length, and K values
            if iteration_found:

                # extract mass and length
                mass_match = re.search(r'Masses:\s*(\d+\.\d+)', line)
                length_match = re.search(r'Lengths:\s*(\d+\.\d+)', line)
                K_match = re.search(r'K:\s*\[\[(.*?)\]\]', line)

                if mass_match:
                    iteration_data['mass'] = float(mass_match.group(1))
                if length_match:
                    iteration_data['length'] = float(length_match.group(1))

                if K_match:
                    # iteration_data['K'] = float(K_match.group(1))
                    K_values = list(map(float, K_match.group(1).split()))
                    iteration_data['K'] = torch.tensor(K_values).reshape(1, 2)  # Assuming K is 1x2 matrix
                
    return iteration_data['mass'], iteration_data['length'], iteration_data['K']



def run_inference_on_model(model, XData, YS, total_time, dt=0.01, context=1, start_index=1, mass = 1, length = 1):
    """
    Runs inference on a trained model to simulate the dynamics of a pendulum system over time, given an initial state and context data.

    Args:
        model (torch.nn.Module): The trained model used for predicting control inputs.
        XData (torch.Tensor): The input state  (e.g., [theta, thetadot])
        YS (torch.Tensor): The ground truth control input data
        total_time (float): Total duration of the simulation in seconds.
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
        context (int, optional): The number of previous time steps used as context for the model. Defaults to 1.
        start_index (int, optional): The starting index for inference. Must be at least equal to `context`. Defaults to 1.
        mass (float, optional): The mass of the pendulum. Defaults to 1.
        length (float, optional): The length of the pendulum. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - T (np.ndarray): Array of time steps during the simulation.
            - theta_model (np.ndarray): Array of predicted theta over time.
            - thetadot_model (np.ndarray): Array of predicted thetadot over time.

    Raises:
        AssertionError: If `start_index` is less than `context`.
    """
    assert start_index >= context, "start_index must be at least equal to context"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    T = np.arange(0, total_time, dt)
    n_steps = len(T)

    XData_context = XData[start_index - context:start_index].to(device)
    YS_context = YS[start_index - context:start_index].to(device)

    # XData_context_copy = XData_context.clone()
    # YS_context_copy = YS_context.clone()
    
    counter = 0
    for i in range(start_index, n_steps):
        with torch.no_grad():
            u_pred = model(XData_context, YS_context, inf = "yes")
            u = u_pred[0][-2].cpu().numpy()  

        theta, thetadot = workCon.single_step_inverted_pendulum_rk4(
            [XData_context[-1][0].cpu().numpy(), XData_context[-1][1].cpu().numpy()],
            u,
            dt, mass = mass, length = length
        )
     
        new_X = torch.tensor([theta, thetadot], dtype=torch.float32, device=device).unsqueeze(0)
        XData_context = torch.cat((XData_context, new_X), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length (sliding window)
        # XData_context_copy = torch.cat((XData_context_copy, new_X), dim=0)
   
        new_Y = torch.tensor(u, dtype=torch.float32, device=device).squeeze() 
       
        if counter == 0:
            YS_context = torch.cat((YS_context[:-1], new_Y.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:-1] to fix the context length
            # YS_context = YS_context[1:] 
            # YS_context_copy = torch.cat((YS_context_copy[:-1], new_Y.unsqueeze(0)), dim=0)
            counter = 1
            # print(YS_context.shape)
        else:
            YS_context = torch.cat((YS_context, new_Y.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length
            # print(YS_context.shape)
            # YS_context_copy = torch.cat((YS_context_copy, new_Y.unsqueeze(0)), dim=0)
            # YS_context = YS_context[1:] ###### 2/11/2025 (ebonye): added this line to fix the context length
  
    theta_model = XData_context[:, 0].cpu().numpy()
    thetadot_model = XData_context[:, 1].cpu().numpy()
    return T, theta_model, thetadot_model


# def load_model(run_dir, name, run_id, step):
def load_model(run_dir, name, run_id, step, epoch):
    """
    Loads a pre-trained model and its configuration from a specified run directory.

    Args:
        run_dir (str): The base directory containing the model runs.
        name (str): The name of the model
        run_id (str): The unique identifier for the specific run to load the model from.
        step (int, optional): The training step at which to load the model checkpoint. 

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded model.
            - conf (dict): The configuration dictionary associated with the model run.
    """
    run_path = os.path.join(run_dir, name, run_id)
    model, conf = get_model_from_run(run_path, epoch= epoch, step=step)
    return model, conf

model_name= "test"
model_run_id= "06a99b9e-ff31-4a4e-a3bf-191df6b0b787"
model_checkpoint_step= 5000 #125000
model_checkpoint_epoch = 2 #50
context = 20
start_index = context


### Load models
models = []
model_steps = []
while True:
    try:
        model, _ = load_model(
        run_dir="./models",
        name= model_name,
        run_id= model_run_id,
        step=model_checkpoint_step,
        epoch=model_checkpoint_epoch)

        models.append(model)
        model_steps.append(model_checkpoint_step)
        model_checkpoint_step += 5000
        model_checkpoint_epoch += 2
        print(f"Model loaded at step {model_checkpoint_step}")
    except:

        break

pends = np.arange(0, 2000, 1)
masses = []
lengths = []
X0s = []
data_and_controls = []
for multipend_num in pends:
    pickle_dir = "dataset_pendulum/picklefolder_test"
    pickle_file = f"multipendulum_test_{multipend_num}.pkl"
    file_path_test_data = os.path.join(pickle_dir, pickle_file)
    with open(file_path_test_data, "rb") as f:
        data = pickle.load(f)
        data_and_controls.append(data)

    file_path_mass_length = "dataset_pendulum/dataset_test_logger.txt"
    masses_temp, lengths_temp, K_values = get_mass_length_Ks_from_text_file(file_path_mass_length, multipend_num)
    masses.append(masses_temp)
    lengths.append(lengths_temp)
    # X0s.append(generate_random_X0())
    X0s.append(np.squeeze(data[0])[0].cpu().detach().numpy())
    print("finished loading pendulum number: ", multipend_num)

X0s_stored = X0s

mse_results = {step : [] for step in model_steps}

for i, model in enumerate(models):
    for j, data_controls in enumerate(data_and_controls):
        X0 = X0s[j]
        mass = masses[j]
        length = lengths[j]
        # XData = data[0]
        YS = data[1]
        total_time = 5
        dt = 0.01

        theta_rk4 = (np.squeeze(data_controls[0]).cpu().detach().numpy())[:, 0]
        thetadot_rk4 = (np.squeeze(data_controls[0]).cpu().detach().numpy())[:, 1]
        control_values_rk4 = (np.squeeze(data_controls[1]).cpu().detach().numpy())
            



        xs_dataset = np.column_stack((theta_rk4, thetadot_rk4))
        xs_dataset = torch.tensor(xs_dataset).float().cuda()
        control_values_rk4 = torch.tensor(control_values_rk4).float().cuda()


        T, theta_model, thetadot_model = run_inference_on_model(model, xs_dataset, control_values_rk4, total_time, dt, context, start_index, mass, length)
        theta_model_temp = theta_model[context:]
        thetadot_model_temp = thetadot_model[context:]
        theta_rk4_temp = theta_rk4[context:]
        thetadot_rk4_temp = thetadot_rk4[context:]
        # mse_val = mse(theta_model, thetadot_model, theta_rk4, thetadot_rk4)
        mse_val = mse(theta_model_temp, thetadot_model_temp, theta_rk4_temp, thetadot_rk4_temp)
        mse_results[model_steps[i]].append(mse_val)
        print(f"Model step: {model_steps[i]}, Pendulum number: {j}, MSE: {mse_val}")


# Save the MSE results
with open("mse_results.pkl", "wb") as f:
    pickle.dump(mse_results, f)






