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



# plot_label = 'multi10_mse_Alternating_diffinitcond_onepend3'
# phase_plot_label = 'multi10_mse_Alternating_sameinitcond_onepend2_phaseplot'
plot_label = 'mse_control'
phase_plot_label = 'mse_control_phaseplot'
mse_plot_label = 'mse_control_mseplot'
save_results = "trainsteps_test_mse_control.txt"
save_phase_plot = "trainsteps_test_mse_control.txt"
log_info = "trainsteps_log_mse_control.txt"
model_name= "test"
model_run_id= "0e78febc-150a-4490-bce8-b6d6d610dc94" #"b18ed057-5e72-472c-a6d8-8da441b57179" #"b30ed57a-7d3e-4be2-8534-57de7bc1508a" #"3712d4a7-2ea1-45db-ba05-4d993e7243ba" #"1376e11b-fa4d-4115-be63-4247b95c4cc0" #"5bb6fb12-1ea0-48d3-b408-998d2e32ca92" #"2289ff4f-8985-4b0a-a68f-fb4bec60a03f" #"11cdf7d5-d01d-4333-85ae-bf5a7465af9e" #"38bf57f0-0a4a-48ed-a423-ea4a38971179" #"9bb50653-5ed4-49c1-8dae-a876b2677236" #"3c33d621-e18a-4c4b-9844-54915b1de7b1"
model_checkpoint_step= 39075 #18400 #46000 #97750 #10800 #29600 #46400 #97750
model_checkpoint_epoch = 25 #59 #125 #14 #38 #60 #125
folder_name = f"inference_run/{plot_label}_{model_checkpoint_step}_{model_run_id}"
mode = 'ood' # 'train', 'ood', 'indistr'

    

total_time = 3 #5 #1.5
dt = 0.01
Num_of_context = 40
Num_of_pendulums = 9 #20 #40 #10




random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1000)

def mse(theta_model, thetadot_model, theta_rk4, thetadot_rk4, device):
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
    xs_pred = torch.tensor(np.column_stack((theta_model, thetadot_model)), dtype=torch.float32, device=device)
    xs_true = torch.tensor(np.column_stack((theta_rk4, thetadot_rk4)), dtype=torch.float32, device=device)
    return (xs_true - xs_pred).pow(2).mean().item()

def mse_controls(control_values_model, control_values_rk4, device):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and true control values.

    Args:
        control_values_model (np.ndarray or list): Model's predicted control values
        control_values_rk4 (np.ndarray or list): True control values using RK4

    Returns:
        float: The computed MSE value, representing the average squared difference between 
        the predicted and true control values.
    """
    control_pred = torch.tensor(control_values_model, dtype=torch.float32, device=device)
    control_true = torch.tensor(control_values_rk4, dtype=torch.float32, device=device)
    return (control_true - control_pred).pow(2).mean().item()

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


def run_inference_on_model(model, XData, YS, total_time, device, dt=0.01, context=1, start_index=1, mass = 1, length = 1):
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
            - YS_context (np.ndarray): Array of control inputs over time.

    Raises:
        AssertionError: If `start_index` is less than `context`.
    """
    assert start_index >= context, "start_index must be at least equal to context"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
  
    T = np.arange(0, total_time, dt)
    n_steps = len(T)

    XData_context = XData[start_index - context:start_index].to(device)
    YS_context = YS[start_index - context:start_index].to(device)

    counter = 0
    for i in range(start_index, n_steps):
        with torch.no_grad():
            u_pred = model(XData_context, YS_context, inf = "yes")
            # u_pred = model(XData_context_scaled, YS_context_scaled, inf = "yes")
            u = u_pred[0][-2].cpu().numpy()  
            u = u * (5*0.98**i)
            # import ipdb; ipdb.set_trace()
            # print(f"u_pred: {u_pred}")
            # print(f"u: {u}")

        theta, thetadot = workCon.single_step_inverted_pendulum_rk4(
            [XData_context[-1][0].cpu().numpy()*(5*0.98**i), XData_context[-1][1].cpu().numpy()*(5*0.98**i)],
            u,
            dt, mass = mass, length = length
        )
     
        new_X = torch.tensor([theta/(5*0.98**i), thetadot/(5*0.98**i)], dtype=torch.float32, device=device).unsqueeze(0)
        XData_context = torch.cat((XData_context, new_X), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length (sliding window)
        # XData_context_copy = torch.cat((XData_context_copy, new_X), dim=0)
        # new_X_scaled = torch.tensor([theta/(5*0.98**i), thetadot/(5*0.98**i)], dtype=torch.float32, device=device).unsqueeze(0)
        # XData_context_scaled = torch.cat((XData_context_scaled, new_X_scaled), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length (sliding window)
   
        new_Y = torch.tensor(u/(5*0.98**i), dtype=torch.float32, device=device).squeeze() 
        # new_Y_scaled = torch.tensor(u/(5*0.98**i), dtype=torch.float32, device=device).squeeze()
       
        if counter == 0:
            YS_context = torch.cat((YS_context[:-1], new_Y.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:-1] to fix the context length
            # YS_context_scaled = torch.cat((YS_context_scaled[:-1], new_Y_scaled.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:-1] to fix the context length
            # YS_context = YS_context[1:] 
            # YS_context_copy = torch.cat((YS_context_copy[:-1], new_Y.unsqueeze(0)), dim=0)
            counter = 1
            # print(YS_context.shape)
        else:
            YS_context = torch.cat((YS_context, new_Y.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length
            # YS_context_scaled = torch.cat((YS_context_scaled, new_Y_scaled.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length
            # print(YS_context.shape)
            # YS_context_copy = torch.cat((YS_context_copy, new_Y.unsqueeze(0)), dim=0)
            # YS_context = YS_context[1:] ###### 2/11/2025 (ebonye): added this line to fix the context length
  
    theta_model = XData_context[:, 0].cpu().numpy()
    thetadot_model = XData_context[:, 1].cpu().numpy()
    YS_context = YS_context.cpu().numpy()

    # import ipdb; ipdb.set_trace()
    # theta_model = [theta_model[i]*(5*0.98**i) for i in range(len(theta_model))]
    theta_model = [theta_model[i-(start_index-context)]*(5*0.98**i) for i in range(start_index-context, n_steps)]
    theta_model = np.stack(theta_model, axis=0)
    # thetadot_model = [thetadot_model[i]*(5*0.98**i) for i in range(len(thetadot_model))]
    thetadot_model = [thetadot_model[i-(start_index-context)]*(5*0.98**i) for i in range(start_index-context, n_steps)]
    thetadot_model = np.stack(thetadot_model, axis=0)
    # YS_context = [YS_context[i]*(5*0.98**i) for i in range(len(YS_context))]
    YS_context = [YS_context[i-(start_index-context)]*(5*0.98**i) for i in range(start_index-context, n_steps-1)]
    YS_context = np.stack(YS_context, axis=0)
    
    # return T, theta_model, thetadot_model
    return T, theta_model, thetadot_model, YS_context

def plot_and_log_results(x_axis, context_lengths, save_results_path, folder_name, plot_label):
    """
    Logs the x-axis and y-axis values to a file and plots the accumulated MSE vs. context length graph.

    Args:
        x_axis (list): The x-axis values (e.g., context lengths).
        context_lengths (list): The y-axis values (e.g., accumulated MSE).
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.

    Returns:
        None
    """
    with open(save_results_path, "a") as f:
        f.write(f"graphs x_axis: {x_axis}\n")
        f.write(f"graphs y_axis: {context_lengths}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, context_lengths, marker='o', label=plot_label)
    plt.xticks(x_axis)
    plt.xlabel('Context Length')
    plt.ylabel('Accumulated MSE')
    plt.title('Accumulated MSE vs Context Length')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(folder_name, f"mse_vs_context_length({plot_label}).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')

def plot_time_series(T, theta_model, thetadot_model, theta_rk4, thetadot_rk4, contextlength, save_results_path, folder_name, plot_label):
    """
    Plots the time series of theta and thetadot.

    Args:
        T (np.ndarray): Array of time steps.
        theta_model (np.ndarray): Array of predicted theta values.
        thetadot_model (np.ndarray): Array of predicted thetadot values.
        theta_rk4 (np.ndarray): Array of true theta values.
        thetadot_rk4 (np.ndarray): Array of true thetadot values.
        context_length (int): The context length used for the inference.
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.

    Returns:
        None
    """
    # with open(save_results_path, "a") as f:
    #     f.write(f"graphs x_axis: {T}\n")
    #     f.write(f"graphs y_axis: {theta_model}\n")
    #     f.write(f"graphs y_axis: {thetadot_model}\n")
    #     f.write(f"graphs y_axis: {theta_rk4}\n")
    #     f.write(f"graphs y_axis: {thetadot_rk4}\n")

    plt.figure(figsize=(10, 6))

    plt.plot(T, theta_model, label='Model Prediction (Theta)')
    plt.plot(T, thetadot_model, label='Model Prediction (ThetaDot)')
    plt.plot(T[contextlength-1], theta_model[contextlength], marker='D', label='ICL Begins (Theta)', color='black')
    plt.plot(T[contextlength-1], thetadot_model[contextlength], marker='D', label='ICL Begins (ThetaDot)', color='red')
    plt.plot(T, theta_rk4, label='RK4 Solver (Theta)')
    plt.plot(T, thetadot_rk4, label='RK4 Solver (ThetaDot)')
    plt.xlabel('Time')
    plt.ylabel('Theta/ThetaDot')
    plt.title(f"Time Series Plot")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(folder_name, f"time_series_plot({plot_label}).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')

def plot_phase_plot(theta_model, thetadot_model, theta_rk4, thetadot_rk4, theta_dmd, thetadot_dmd, context_length, save_results_path, folder_name, plot_label):
    """
    Plots the phase plot of theta vs. thetadot.

    Args:
        theta (np.ndarray): Array of theta values.
        thetadot (np.ndarray): Array of thetadot values.
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.

    Returns:
        None
    """
    # with open(save_results_path, "a") as f:
    #     f.write(f"graphs x_axis: {theta}\n")
    #     f.write(f"graphs y_axis: {thetadot}\n")

    # print(f"theta_model: {theta_model}")
    # print(f"thetadot_model: {thetadot_model}")
    plt.figure(figsize=(10, 6))
    plt.plot(theta_model, thetadot_model, marker='s', color='orange', label='Model Prediction')
    # plt.plot(theta_dmd, thetadot_dmd, marker='^', color='red', label='Dynamic Mode Decomposition')
    plt.plot(theta_rk4, thetadot_rk4, marker='x', color='green', label='RK4 Solver')
    

    plt.plot(theta_model[context_length], thetadot_model[context_length], marker = 'D', label='ICL Begins', color='black')
    plt.xlabel('Theta')
    plt.ylabel('ThetaDot')
    plt.title(f"Phase Plot (Context Length: {context_length})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(folder_name, f"phase_plot({plot_label}).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')


def plot_mse_vs_context_length(mean_mse, std_mse, save_results_path, folder_name, plot_label):
    """
    Plots the mean MSE vs. context length graph.

    Args:
        mean_mse (dict): A dictionary containing the mean MSE values for each context length.
        std_mse (dict): A dictionary containing the standard deviation of MSE values for each context length.
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.

    Returns:
        None
    """
    # with open(save_results_path, "a") as f:
    #     f.write(f"mean_mse: {mean_mse}\n")
    #     f.write(f"std_mse: {std_mse}\n")

    x_axis = list(mean_mse.keys())
    y_axis = list(mean_mse.values())
    y_err = list(std_mse.values())

    plt.figure(figsize=(10, 6))
    plt.errorbar(x_axis, y_axis, yerr=y_err, fmt='o-', capsize=8, label=plot_label)
    plt.xticks(x_axis)
    plt.xlabel('Context Length')
    plt.ylabel('Mean MSE')
    plt.title(f'Mean MSE vs Context Length {Num_of_pendulums} Pendulums')
    plt.legend()
    plt.grid(True)


    plot_path = os.path.join(folder_name, f"mean_mse_vs_context_length({Num_of_pendulums} Pendulums).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')


def dynamic_mode_decomposition(XData, X0, total_time, dt, context):
        if context == 1:
            ######## does not work well for one context
            # Single snapshot, apply pseudo-DMD directly
            X = XData.cpu().detach().numpy() if hasattr(XData, 'cpu') else XData
            X = X.T
            Y = X

            # Regularize the inverse process (pseudo-DMD) for one snapshot
            X_pseudo = (np.linalg.pinv(X.T @ X + 1e-6 * np.eye(X.shape[1])) @ X.T).T
            # print(np.shape(X_pseudo))
            U, S, V = np.linalg.svd(X_pseudo, full_matrices=False)
            # print(np.shape(U))
            # print(np.shape(S))
            # print(np.shape(V))
            # print(np.shape(X))
            # print(np.shape(Y))
            Atilde = U.T @ Y @ V.T @ np.linalg.inv(np.diag(S))
            eigvals, eigvecs = np.linalg.eig(Atilde)
            # print(f'Eigenvalues:{eigvals}')
            # print(np.shape(Atilde))
            # Phi = Y @ V.T @ np.linalg.inv(np.diag(S)) @ eigvecs
            # Phi = eigvecs
            # print(np.shape(Phi))

        else:
            X_trunc = XData[0:context-2+1]
            Y_trunc = XData[1:context-1+1]
            X = X_trunc.T
            Y = Y_trunc.T
            X = X.cpu().detach().numpy() if hasattr(X, 'cpu') else X
            Y = Y.cpu().detach().numpy() if hasattr(Y, 'cpu') else Y


            U, S, V = np.linalg.svd(X, full_matrices=False)
            # print(f'Context: {context}')
            # print(f'X_trunc: {X_trunc}')
            # print(f'Y_trunc: {Y_trunc}')
            # print(f'X: {X}')
            # print(f'Y: {Y}')
            # print(f'U: {U}')
            # print(f'S: {S}')
            # print(f'V: {V}')

            # A = np.linalg.multi_dot([Y, V.T, np.linalg.inv(np.diag(S)) , U.T])
            Atilde = U.T @ Y @ V.T @ np.linalg.inv(np.diag(S))

            eigvals, eigvecs = np.linalg.eig(Atilde)
            # print(f'Atilde: {Atilde}')
            # print(f'Eigenvalues of Atilde:{eigvals}')
            # print(f'Eigenvecs of Atilde:{eigvecs}')
        
        Phi = Y @ V.T @ np.linalg.inv(np.diag(S)) @ eigvecs
        # b = np.linalg.pinv(Phi) @ X[:, 0]
        b = np.linalg.pinv(Phi) @ X0

        Omega = np.log(eigvals) / dt
        # print(f'Omega: {Omega}')
        # omega = np.log(eigvals) / dt
        # Phi = np.linalg.multi_dot([XData[1:context-1].T, V, np.linalg.inv(np.diag(S)), U.T, eigvecs])
        # b = np.linalg.lstsq(Phi, XData[1:context-1].T @ A.T)[0]

        T = np.arange(0, total_time, dt)
        Tnew = T[context:]
        X_dmd = np.zeros((len(Phi), len(T)), dtype=np.complex128)
        X_dmd[:, 0:context] = (XData[0:context].cpu().detach().numpy() if hasattr(XData, 'cpu') else XData[0:context]).T
        for i,t in enumerate(Tnew):
            X_dmd[:,i+context] = (Phi @ (b*np.exp(Omega*t))).real

        theta_dmd = X_dmd[0, :].real
        thetadot_dmd = X_dmd[1, :].real
        
        return theta_dmd, thetadot_dmd

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



def main():
    """_summary_
    """
    model, _ = load_model(
        run_dir="./models",
        name= model_name,
        run_id= model_run_id,
        step=model_checkpoint_step,
        epoch=model_checkpoint_epoch ###### 2/11/2025 (ebonye): added epoch
    )

    os.makedirs(folder_name, exist_ok=True)
    save_results_path = os.path.join(folder_name, save_results)
    save_phase_path = os.path.join(folder_name, save_phase_plot)
    log_info_path = os.path.join(folder_name, log_info)
    context_lengths = [0] * Num_of_context
    # start_indices = [Num_of_context]
    contexts = np.arange(1, Num_of_context + 1)

    ######
    # masses = [generate_dataset.sample_mass_uniform() for _ in range(Num_of_pendulums)]
    # lengths = [generate_dataset.sample_length_uniform() for _ in range(Num_of_pendulums)]

    ###Training point
    # masses = [0.1303369478303672]
    # lengths = [0.20875376432793344]
    # multipend_num = 10
    # pickle_dir = "dataset_pendulum/picklefolder"
    # pickle_file = f"multipendulum_{multipend_num}.pkl"
    # file_path_train_data = os.path.join(pickle_dir, pickle_file)
    # with open(file_path_train_data, "rb") as f:
    #     data_controls = pickle.load(f)

    # file_path_mass_length = "dataset_pendulum/dataset_logger_noreseeding.txt"

    ###Test point
    # multipend_num = 1000
    # pickle_dir = "dataset_pendulum/picklefolder_test"
    # pickle_file = f"multipendulum_test_{multipend_num}.pkl"
    # file_path_test_data = os.path.join(pickle_dir, pickle_file)
    # with open(file_path_test_data, "rb") as f:
    #     data_controls = pickle.load(f)

    # file_path_mass_length = "dataset_pendulum/dataset_test_logger.txt"


    ####################
    # masses, lengths, K_values = get_mass_length_Ks_from_text_file(file_path_mass_length, multipend_num)
    # print(f"masses: {masses}")
    # print(f"lengths: {lengths}")

    # # masses= [0.1567184391040723]
    # # lengths= [0.3907888188883454]

    # masses = [masses]
    # lengths = [lengths]

    ####################
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    pends = np.random.randint(0, 5000, size=Num_of_pendulums)
    # pends = [1050] 
    masses = []
    lengths = []
    X0s = []
    data_and_controls = []
    for multipend_num in pends:
        if mode == 'ood':
            pickle_dir = f"dataset_pendulum/picklefolder_test_outofdistr"
            pickle_file = f"multipendulum_test_outofdistr_{multipend_num}.pkl"
            file_path_mass_length = f"dataset_pendulum/dataset_test_outofdistr_logger.txt"
        elif mode == 'train':
            pickle_dir = f"dataset_pendulum/picklefolder"
            pickle_file = f"multipendulum_{multipend_num}.pkl"
            file_path_mass_length = f"dataset_pendulum/dataset_logger_train.txt"
        elif mode == 'indistr':
            pickle_dir = f"dataset_pendulum/picklefolder_test_indistr"
            pickle_file = f"multipendulum_test_{multipend_num}.pkl"
            file_path_mass_length = f"dataset_pendulum/dataset_test_logger.txt"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        
        file_path_test_data = os.path.join(pickle_dir, pickle_file)
        with open(file_path_test_data, "rb") as f:
            data = pickle.load(f)
            data_and_controls.append(data)

       
        masses_temp, lengths_temp, K_values = get_mass_length_Ks_from_text_file(file_path_mass_length, multipend_num)
        masses.append(masses_temp)
        lengths.append(lengths_temp)
        X0s.append(np.squeeze(data[0])[0].cpu().detach().numpy())

    X0s_stored = X0s

    data_and_controls_unscaled = []
    for dta_ctrl in data_and_controls:
        theta_rk4 = (np.squeeze(dta_ctrl[0]).cpu().detach().numpy())[:, 0]
        thetadot_rk4 = (np.squeeze(dta_ctrl[0]).cpu().detach().numpy())[:, 1]
        control_values_rk4 = (np.squeeze(dta_ctrl[1]).cpu().detach().numpy())
        theta_rk4_unscaled = np.stack([theta_rk4[i]*(5*0.98**i) for i in range(len(theta_rk4))], axis=0)
        thetadot_rk4_unscaled = np.stack([thetadot_rk4[i]*(5*0.98**i) for i in range(len(thetadot_rk4))], axis=0)
        control_values_rk4_unscaled = np.stack([control_values_rk4[i]*(5*0.98**i) for i in range(len(control_values_rk4))], axis=0)
        state = np.column_stack((theta_rk4_unscaled, thetadot_rk4_unscaled))
        # data_and_controls_unscaled.append(np.stack([state, control_values_rk4_unscaled], axis=1))
        data_and_controls_unscaled.append((torch.tensor(state).float().cuda(), torch.tensor(control_values_rk4_unscaled).float().cuda()))
    
    
    with open(log_info_path, "w") as file:
        mse_results = {context_length: [] for context_length in contexts}
        mse_control_results = {context_length: [] for context_length in contexts}
        phase_data = {context_length: [] for context_length in contexts}
        controls_data = {context_length: [] for context_length in contexts}
        counter = 0
        for mass, length in tqdm(zip(masses, lengths), desc="MultiPendulum", total=len(masses), leave=False):
            # X0 = generate_random_X0()
            # X0 = [ 1.0852e+00, -1.3760e+00]
            # X0 = [1.4575, 0.1397]
            mse_per_context = []
            mse_control_per_context = []
            # X0 = np.squeeze(data_controls[0])[0].cpu().detach().numpy()
            # print(f"X0: {X0}")
            # X0 = X0s.pop(0) 

            


            # T1, theta_rk4, thetadot_rk4, control_values_rk4, K_values = workCon.checking(
            #         X0, total_time, method='rk4', dt=dt, mass = mass, length = length
            #     )

            data_controls = data_and_controls[counter]
            theta_rk4 = (np.squeeze(data_controls[0]).cpu().detach().numpy())[:, 0]
            thetadot_rk4 = (np.squeeze(data_controls[0]).cpu().detach().numpy())[:, 1]
            control_values_rk4 = (np.squeeze(data_controls[1]).cpu().detach().numpy())

            data_controls_unscaled = data_and_controls_unscaled[counter]
            theta_rk4_unscaled = (np.squeeze(data_controls_unscaled[0]).cpu().detach().numpy())[:, 0]
            thetadot_rk4_unscaled = (np.squeeze(data_controls_unscaled[0]).cpu().detach().numpy())[:, 1]
            control_values_rk4_unscaled = (np.squeeze(data_controls_unscaled[1]).cpu().detach().numpy())

            
            

            # file.write(f"Current pendulum mass: {mass}\n")
            # file.write(f"Current pendulum length: {length}\n")
            # file.write(f"X0: {X0}\n")
            # # file.write(f"T1 (RK4 Time): {T1}\n")
            # file.write(f"K_values (RK4 K_values): {K_values}\n\n")


            xs_dataset = np.column_stack((theta_rk4, thetadot_rk4))
            xs_dataset = torch.tensor(xs_dataset).float().cuda()
            control_values_rk4 = torch.tensor(control_values_rk4).float().cuda()
            # store_theta_model = []
            # store_thetadot_model = []
            # for start_index in start_indices:
            # for context in contexts:
            start_index = Num_of_context
            for context in tqdm(contexts, desc="Context Loop", leave=False):
                # file.write(f"  Start Index: {start_index}\n")
                # theta_rk4_temp = theta_rk4[start_index-1:]
                # thetadot_rk4_temp = thetadot_rk4[start_index-1:]

                theta_rk4_temp = theta_rk4[context:]
                thetadot_rk4_temp = thetadot_rk4[context:]
                controls_rk4_temp = control_values_rk4[context:]

                # T_model, theta_model2, thetadot_model2, controls_model2 = run_inference_on_model(
                #     model, xs_dataset, control_values_rk4, total_time, device, dt, context=context, start_index=context, mass = mass, length = length
                # )

                T_model, theta_model2, thetadot_model2, controls_model2 = run_inference_on_model(
                    model, xs_dataset, control_values_rk4, total_time, device, dt, context=context, start_index=start_index, mass = mass, length = length
                )                

                
                # store_theta_model.append(theta_model2)
                # store_thetadot_model.append(thetadot_model2)

                trajectory = np.stack([theta_model2, thetadot_model2], axis=1)
                controls_for_trajectory = controls_model2
                phase_data[context].append(trajectory)
                controls_data[context].append(controls_for_trajectory)

                # theta_model = theta_model2[context:]
                # thetadot_model = thetadot_model2[context:]
                # controls_model = controls_model2[context-1:]

                theta_model = theta_model2[start_index:]
                thetadot_model = thetadot_model2[start_index:]
                controls_model = controls_model2[start_index-1:]
                # control_model = control_values_rk4[context:]
                # mse_loss = mse(theta_model, thetadot_model, theta_rk4_temp, thetadot_rk4_temp, device)
                # mse_loss = mse(theta_model, thetadot_model, theta_rk4_unscaled[context:], thetadot_rk4_unscaled[context:], device)

                # import ipdb; ipdb.set_trace()
                mse_loss = mse(theta_model2, thetadot_model2, theta_rk4_unscaled[start_index-context:], thetadot_rk4_unscaled[start_index-context:], device)
                mse_per_context.append(mse_loss)

                # mse_control_loss = mse_controls(controls_model, controls_rk4_temp, device)
                # mse_control_loss = mse_controls(controls_model2, control_values_rk4_unscaled[context:], device)
                mse_control_loss = mse_controls(controls_model2, control_values_rk4_unscaled[start_index-context+1:], device)
                mse_control_per_context.append(mse_control_loss)

                


            for idx, context_length in enumerate(contexts):
                mse_results[context_length].append(mse_per_context[idx])
                mse_control_results[context_length].append(mse_control_per_context[idx])
            counter += 1
            
        mse_mean = {context_length: np.mean(mse_results[context_length]) for context_length in contexts}
        mse_std = {context_length: np.std(mse_results[context_length]) for context_length in contexts}
        # print(f"Mean MSE: {mse_mean}")
        # print(f"Std MSE: {mse_std}")
        mse_control_mean = {context_length: np.mean(mse_control_results[context_length]) for context_length in contexts}
        mse_control_std = {context_length: np.std(mse_control_results[context_length]) for context_length in contexts}

    # plot_mse_vs_context_length(mse_mean, mse_std, save_results_path, folder_name, mse_plot_label)
    plot_mse_vs_context_length(mse_control_mean, mse_control_std, save_results_path, folder_name, mse_plot_label)


                # for context1 in tqdm(range(len(context_lengths)), desc=f"Context Loop (Start Index {start_index})", leave=False):
                #     if start_index < context1:
                #         continue
                #     # print(f"Context1: {context1 + 1}")
                #     T_model, theta_model2, thetadot_model2 = run_inference_on_model(
                #             model, xs_dataset, control_values_rk4, total_time, dt, context=context1 + 1, start_index=start_index, mass = mass, length = length
                #         )
                #     store_theta_model.append(theta_model2)
                #     store_thetadot_model.append(thetadot_model2)
                #     theta_model = theta_model2[context1:]
                #     thetadot_model = thetadot_model2[context1:]
                #     # print(np.shape(theta_model2))
                #     # print(np.shape(theta_model))
                #     # print(np.shape(thetadot_model))
                #     # print(np.shape(theta_rk4_temp))
                #     # print(np.shape(thetadot_rk4_temp))
                #     mse_loss = mse(theta_model, thetadot_model, theta_rk4_temp, thetadot_rk4_temp)


                #     file.write(f"    Context1: {context1 + 1}\n")
                #     file.write(f"    Before Splice Theta Model: {theta_model2.tolist()}\n")
                #     file.write(f"    Before Splice Thetadot Model: {thetadot_model2.tolist()}\n")
                #     file.write(f"    Before Splice theta_rk4_temp: {theta_rk4.tolist()}\n")
                #     file.write(f"    Before Splice thetadot_rk4_temp: {thetadot_rk4.tolist()}\n")
                #     file.write(f"    After Splice Theta Model: {theta_model.tolist()}\n")
                #     file.write(f"    After Splice theta_rk4_temp: {theta_rk4_temp.tolist()}\n")
                #     file.write(f"    After Splice Thetadot Model: {thetadot_model.tolist()}\n")
                #     file.write(f"    After Splice thetadot_rk4_temp: {thetadot_rk4_temp.tolist()}\n")
                #     file.write(f"    MSE Loss: {mse_loss}\n")
                #     file.write(f"    Context Lengths (Accum MSE): {context_lengths}\n\n")

                    
                #     context_lengths[context1] += mse_loss
                #     if context1 == Num_of_context - 1:
                #         # theta_full_model = np.hstack((theta_model2,))
                #         # print(np.shape(T_model))    
                #         # print(np.shape(theta_model2))
                #         # print(np.shape(thetadot_model2))
                #         # print(xs_dataset[start_index-(context1+1):start_index])
                #         theta_dmd, thetadot_dmd = dynamic_mode_decomposition(xs_dataset[start_index-(context1+1):start_index], X0, total_time, dt, context1+1)
                #         # plot_phase_plot(theta_model2, thetadot_model2, theta_rk4, thetadot_rk4, theta_ivp, thetadot_ivp, theta_rk4, thetadot_rk4, context1+1, save_results_path, folder_name, plot_label)
                #         plot_phase_plot(theta_model2, thetadot_model2, theta_rk4, thetadot_rk4, theta_dmd, thetadot_dmd, context1+1, save_results_path, folder_name, plot_label)
                #         # print(np.shape(theta_model2))
                #         # print(np.shape(T_model))
                #         # print(np.shape(theta_rk4))
                #         plot_time_series(T_model, theta_model2, thetadot_model2, theta_rk4, thetadot_rk4, context1+1, save_results_path, folder_name, plot_label)
                        

    # x_axis = list(range(1, len(context_lengths) + 1))
    # plot_and_log_results(x_axis, context_lengths, save_results_path, folder_name, phase_plot_label)

    # return X0, masses, lengths, store_theta_model, store_thetadot_model
    return X0s_stored, masses, lengths, phase_data, controls_data, data_and_controls_unscaled, pends

try:
    results = main()
    # print(np.array(theta_models).shape)
    # print(np.array(thetadot_models).shape)

    # save_results = os.join(folder_name, "results.pkl")
    save_results = os.path.join(folder_name, f"results_maxcontext{Num_of_context}_numpends{Num_of_pendulums}_{mode}_changestart.pkl")

    with open(save_results, "wb") as f:
        pickle.dump(results, f)


    print("done")   
except Exception:
    print("exception starting debugger")
    traceback.print_exc()
    ipdb.post_mortem()
