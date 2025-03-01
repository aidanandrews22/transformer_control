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



# plot_label = 'multi10_mse_Alternating_diffinitcond_onepend3'
# phase_plot_label = 'multi10_mse_Alternating_sameinitcond_onepend2_phaseplot'
plot_label = 'onepend_mse_sameinitcond6'
phase_plot_label = 'onepend_mse_sameinitcond6_phaseplot'
save_results = "trainsteps100000_test1_sameinitcond6_onepend.txt"
save_phase_plot = "trainsteps100000_test1_sameinitcond6_onepend.txt"
log_info = "trainsteps100000_log1_diff_initcond6_onepend.txt"
model_name= "test"
model_run_id= "20a0c7c3-0dc1-4333-a975-9561640305e6" #"9bbb6dd7-4ca0-48f5-85fa-e246a773414d" #"20a0c7c3-0dc1-4333-a975-9561640305e6" #"945102bf-6f1d-487a-a829-0686f27a54d0" #(q matrix diag(4,1))"4c4aa2ff-7006-4234-8eff-7a9a727efc1a" #(q matrix diag(10,1))"b222c1e0-3b61-4719-9d54-f6252bdb0e0d" 
model_checkpoint_step= 125000 #70000 #125000 #68000 #40000 #275000 #200 #100000 #80000 #200 #600
folder_name = f"inference_run/{plot_label}_{model_checkpoint_step}"

    

total_time = 5 #1.5
dt = 0.01
Num_of_context = 5
Num_of_pendulums = 1 #40 #10




random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1000)

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

def load_model(run_dir, name, run_id, step):
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
    model, conf = get_model_from_run(run_path, step=step)
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
  
    T = np.arange(0, total_time + dt, dt)
    n_steps = len(T)

    XData_context = XData[start_index - context:start_index].to(device)
    YS_context = YS[start_index - context:start_index].to(device)

    XData_context_copy = XData_context.clone()
    YS_context_copy = YS_context.clone()
    
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
        XData_context_copy = torch.cat((XData_context_copy, new_X), dim=0)
   
        new_Y = torch.tensor(u, dtype=torch.float32, device=device).squeeze() 
       
        if counter == 0:
            YS_context = torch.cat((YS_context[:-1], new_Y.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:-1] to fix the context length
            # YS_context = YS_context[1:] 
            YS_context_copy = torch.cat((YS_context_copy[:-1], new_Y.unsqueeze(0)), dim=0)
            counter = 1
            # print(YS_context.shape)
        else:
            YS_context = torch.cat((YS_context, new_Y.unsqueeze(0)), dim=0) ###### 2/11/2025 (ebonye): added [1:] to fix the context length
            # print(YS_context.shape)
            YS_context_copy = torch.cat((YS_context_copy, new_Y.unsqueeze(0)), dim=0)
            # YS_context = YS_context[1:] ###### 2/11/2025 (ebonye): added this line to fix the context length
  
    theta_model = XData_context_copy[:, 0].cpu().numpy()
    thetadot_model = XData_context_copy[:, 1].cpu().numpy()
    return T, theta_model, thetadot_model

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
    plt.plot(T[contextlength-1], theta_rk4[contextlength-1], marker='D', label='ICL Begins (Theta)', color='black')
    plt.plot(T[contextlength-1], thetadot_rk4[contextlength-1], marker='D', label='ICL Begins (ThetaDot)', color='red')
    plt.plot(T, theta_rk4, label='RK4 Solver (Theta)')
    plt.plot(T, thetadot_rk4, label='RK4 Solver (ThetaDot)')
    plt.xlabel('Time')
    plt.ylabel('Theta/ThetaDot')
    plt.title(f"Time Series Plot")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(folder_name, f"time_series_plot({plot_label}).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')

def plot_phase_plot(theta_model, thetadot_model, theta_rk4, thetadot_rk4, theta_ivp, thetadot_ivp, theta_dmd, thetadot_dmd, context_length, save_results_path, folder_name, plot_label):
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
    plt.plot(theta_dmd, thetadot_dmd, marker='^', color='red', label='Dynamic Mode Decomposition')
    plt.plot(theta_rk4, thetadot_rk4, marker='x', color='green', label='RK4 Solver')
    # plt.plot(theta_ivp[:-context_length], thetadot_ivp[:-context_length], marker= '*', color='blue', label='Ivp Solver', alpha=0.5)
    
    plt.plot(theta_rk4[context_length-1], thetadot_rk4[context_length-1], marker = 'D', label='ICL Begins', color='black')
    plt.xlabel('Theta')
    plt.ylabel('ThetaDot')
    plt.title(f"Phase Plot (Context Length: {context_length})")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(folder_name, f"phase_plot({plot_label}).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')


def plot_using_ivpsolver(X0, total_time, dt, mass, length):
    """
    Plots the phase plot of theta vs. thetadot using the IVP solver.
    
    Args:
        X0 (list): Initial state of the pendulum [theta, thetadot].
        total_time (float): Total duration of the simulation in seconds.
        dt (float): Time step for the simulation.
        mass (float): Mass of the pendulum.
        length (float): Length of the pendulum.
    """
    T = np.arange(0, total_time + dt, dt)
    n_steps = len(T)


    # x = np.array(X0, dtype=float)
    # theta = np.zeros(n_steps)
    # thetadot = np.zeros(n_steps)
    # tau = np.zeros(n_steps)

    # theta_d = 0.0  # desired state
    # thdot_d = 0.0  # desired state
    X0 = np.array(X0)
    _,_,_,_,K = workCon.checking(X0, total_time, method='rk4', dt=dt, mass = mass, length = length)
    K = np.squeeze(K)
    def f(t,x):
        b = 0.5
        g = 9.81
        x = np.array(x)
        theta = x[0]
        thetadot = x[1]
        dtheta = thetadot
        dthetadot = (-b * thetadot + mass * g * length * np.sin(theta) + (-K @ x)) / (mass * length**2)
        
        return np.array([dtheta, dthetadot])
    
    solve = solve_ivp(f, [0, total_time], X0, t_eval=T)
    theta = solve.y[0]
    thetadot = solve.y[1]

    return [theta, thetadot]

def dynamic_mode_decomposition(XData, X0, total_time, dt, context):
        X_trunc = XData[0:context-2]
        Y_trunc = XData[1:context-1]
        X = X_trunc.T
        Y = Y_trunc.T
        X = X.cpu().detach().numpy()
        Y = Y.cpu().detach().numpy()

        U, S, V = np.linalg.svd(X, full_matrices=False)
        # A = np.linalg.multi_dot([Y, V.T, np.linalg.inv(np.diag(S)) , U.T])
        Atilde = U.T @ Y @ V.T @ np.linalg.inv(np.diag(S))
        eigvals, eigvecs = np.linalg.eig(Atilde)
        Phi = Y @ V.T @ np.linalg.inv(np.diag(S)) @ eigvecs
        # b = np.linalg.pinv(Phi) @ X[:, 0]
        b = np.linalg.pinv(Phi) @ X0

        Omega = np.log(eigvals) / dt
        # omega = np.log(eigvals) / dt
        # Phi = np.linalg.multi_dot([XData[1:context-1].T, V, np.linalg.inv(np.diag(S)), U.T, eigvecs])
        # b = np.linalg.lstsq(Phi, XData[1:context-1].T @ A.T)[0]

        T = np.arange(0, total_time + dt, dt)
        X_dmd = np.zeros((len(Phi), len(T)), dtype=np.complex128)
        for i,t in enumerate(T):
            X_dmd[:,i] = (Phi @ (b*np.exp(Omega*t))).real

        theta_dmd = X_dmd[0, :].real
        thetadot_dmd = X_dmd[1, :].real
        
        return theta_dmd, thetadot_dmd


def main():
    """_summary_
    """
    model, _ = load_model(
        run_dir="./models",
        name= model_name,
        run_id= model_run_id,
        step=model_checkpoint_step
    )

    os.makedirs(folder_name, exist_ok=True)
    save_results_path = os.path.join(folder_name, save_results)
    save_phase_path = os.path.join(folder_name, save_phase_plot)
    log_info_path = os.path.join(folder_name, log_info)
    context_lengths = [0] * Num_of_context
    start_indices = [Num_of_context]

    # masses = [generate_dataset.sample_bounded_gaussian() for _ in range(Num_of_pendulums)]
    # lengths = [generate_dataset.sample_bounded_gaussian() for _ in range(Num_of_pendulums)]
    ######
    # masses = [4.288184753155463] ###### 2/6/2025 (ebonye): trained on this point gaussian
    # lengths = [4.4494456086997705] ###### 2/6/2025 (ebonye): trained on this point gaussian
    ######
    masses = [generate_dataset.sample_mass_uniform() for _ in range(Num_of_pendulums)]
    lengths = [generate_dataset.sample_length_uniform() for _ in range(Num_of_pendulums)]
    ######
    # masses = [0.08535511797882313] ###### 2/6/2025 (ebonye): trained on this point uniform
    # lengths = [0.3293812973204395]
    ######
    # masses = [8] ###### 2/9/2025 (ebonye): way outside scope of trained data and doesn't perform well uniform
    # lengths = [16]
    # masses= [0.09883065955953545]
    # lengths= [0.3418819949803681]

    # masses= [0.0929533105933265]
    # lengths= [0.4086680167523326]

    # masses = [0.11393306704511483]
    # lengths = [0.26983731488107]

    # masses = [0.18]
    # lengths = [0.57]

    # masses = [0.27]
    # lengths = [0.67]

    # masses = [np.random.uniform(0.06, 0.17) for _ in range(Num_of_pendulums)]
    # lengths = [np.random.uniform(0.2, 0.55) for _ in range(Num_of_pendulums)]
    
    with open(log_info_path, "w") as file:
        for mass, length in tqdm(zip(masses, lengths), desc="MultiPendulum", total=len(masses), leave=False):
            X0 = generate_random_X0()
            # X0 = [np.pi/3,1.0]
            # X0 = [-0.0811,  1.3219]
            # X0 = [-0.08689436, 1.321947]
            # X0 = [0.46671834184573213, -2.639731918107688]
            # X0 = [np.pi/2, 3]
            # X0 = [np.pi/2, 3]
            # X0 = [3*np.pi/4, 5]


            T1, theta_rk4, thetadot_rk4, control_values_rk4, K_values = workCon.checking(
                    X0, total_time, method='rk4', dt=dt, mass = mass, length = length
                )
            

            file.write(f"Current pendulum mass: {mass}\n")
            file.write(f"Current pendulum length: {length}\n")
            file.write(f"X0: {X0}\n")
            file.write(f"T1 (RK4 Time): {T1}\n")
            file.write(f"K_values (RK4 K_values): {K_values}\n\n")


            xs_dataset = np.column_stack((theta_rk4, thetadot_rk4))
            xs_dataset = torch.tensor(xs_dataset).float().cuda()
            control_values_rk4 = torch.tensor(control_values_rk4).float().cuda()
            store_theta_model = []
            store_thetadot_model = []
            for start_index in start_indices:
                file.write(f"  Start Index: {start_index}\n")
                theta_rk4_temp = theta_rk4[start_index - 1:]
                thetadot_rk4_temp = thetadot_rk4[start_index - 1:]
                for context1 in tqdm(range(len(context_lengths)), desc=f"Context Loop (Start Index {start_index})", leave=False):
                    if start_index < context1:
                        continue
                    # print(f"Context1: {context1 + 1}")
                    T_model, theta_model2, thetadot_model2 = run_inference_on_model(
                            model, xs_dataset, control_values_rk4, total_time, dt, context=context1 + 1, start_index=start_index, mass = mass, length = length
                        )
                    store_theta_model.append(theta_model2)
                    store_thetadot_model.append(thetadot_model2)
                    theta_model = theta_model2[context1:]
                    thetadot_model = thetadot_model2[context1:]
                    mse_loss = mse(theta_model, thetadot_model, theta_rk4_temp, thetadot_rk4_temp)


                    file.write(f"    Context1: {context1 + 1}\n")
                    file.write(f"    Before Splice Theta Model: {theta_model2.tolist()}\n")
                    file.write(f"    Before Splice Thetadot Model: {thetadot_model2.tolist()}\n")
                    file.write(f"    Before Splice theta_rk4_temp: {theta_rk4.tolist()}\n")
                    file.write(f"    Before Splice thetadot_rk4_temp: {thetadot_rk4.tolist()}\n")
                    file.write(f"    After Splice Theta Model: {theta_model.tolist()}\n")
                    file.write(f"    After Splice theta_rk4_temp: {theta_rk4_temp.tolist()}\n")
                    file.write(f"    After Splice Thetadot Model: {thetadot_model.tolist()}\n")
                    file.write(f"    After Splice thetadot_rk4_temp: {thetadot_rk4_temp.tolist()}\n")
                    file.write(f"    MSE Loss: {mse_loss}\n")
                    file.write(f"    Context Lengths (Accum MSE): {context_lengths}\n\n")

                    
                    context_lengths[context1] += mse_loss
                    if context1 == Num_of_context - 1:
                        # theta_full_model = np.hstack((theta_model2,))
                        print(np.shape(T_model))    
                        print(np.shape(theta_model2))
                        print(np.shape(thetadot_model2))
                        theta_ivp, thetadot_ivp = plot_using_ivpsolver([theta_rk4[0],thetadot_rk4[0]], total_time, dt, mass, length)
                        # theta_dmd, thetadot_dmd = dynamic_mode_decomposition(xs_dataset, X0, total_time, dt, context1)
                        plot_phase_plot(theta_model2, thetadot_model2, theta_rk4, thetadot_rk4, theta_ivp, thetadot_ivp, theta_rk4, thetadot_rk4, context1+1, save_results_path, folder_name, plot_label)
                        plot_time_series(T_model, theta_model2, thetadot_model2, theta_rk4, thetadot_rk4, context1+1, save_results_path, folder_name, plot_label)
                        # print(len(theta_model))
                        # print(len(theta_rk4))
                        # print(len(theta_ivp))
                        # print(len(theta_dmd))
                        # print(f"theta_model: {theta_model}")
                        # print(f"thetadot_model: {thetadot_model}")

    x_axis = list(range(1, len(context_lengths) + 1))
    plot_and_log_results(x_axis, context_lengths, save_results_path, folder_name, phase_plot_label)

    return X0, masses, lengths, store_theta_model, store_thetadot_model

try:
    results = main()
    # print(np.array(theta_models).shape)
    # print(np.array(thetadot_models).shape)

    # save_results = os.join(folder_name, "results.pkl")
    save_results = os.path.join(folder_name, f"results_maxcontext{Num_of_context}.pkl")

    with open(save_results, "wb") as f:
        pickle.dump(results, f)


    print("done")   
except Exception:
    print("exception starting debugger")
    traceback.print_exc()
    ipdb.post_mortem()
