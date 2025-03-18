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

    # theta = np.random.uniform(-np.pi/4, np.pi/4) ######2/8/2025 (ebonye): thirty degree recommended by gpt
    theta = np.random.uniform(np.pi/5, np.pi/2)
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


# def dynamic_mode_decomposition(XData, X0, total_time, dt, context):
#         X_trunc = XData[0:context]
#         Y_trunc = XData[1:context+1]
#         X = X_trunc.T
#         Y = Y_trunc.T
#         X = X.cpu().detach().numpy()
#         Y = Y.cpu().detach().numpy()


#         U, S, V = np.linalg.svd(X, full_matrices=False)

#         ###### 2/17/2025 (ebonye): note V.T is really V
#         Atilde = U.T @ Y @ V.T @ np.linalg.inv(np.diag(S))
#         eigvals, eigvecs = np.linalg.eig(Atilde)
#         Phi = Y @ V.T @ np.linalg.inv(np.diag(S)) @ eigvecs
#         # b = np.linalg.pinv(Phi) @ X[:, 0]
#         b = np.linalg.pinv(Phi) @ X0

#         Omega = np.log(eigvals) / dt
#         # omega = np.log(eigvals) / dt
#         # Phi = np.linalg.multi_dot([XData[1:context-1].T, V, np.linalg.inv(np.diag(S)), U.T, eigvecs])
#         # b = np.linalg.lstsq(Phi, XData[1:context-1].T @ A.T)[0]

#         T = np.arange(0, total_time + dt, dt)
#         X_dmd = np.zeros((len(Phi), len(T)), dtype=np.complex128)
#         for i,t in enumerate(T):
#             X_dmd[:,i] = (Phi @ (np.exp(Omega*t)*b)).real

#         theta_dmd = X_dmd[0, :].real
#         thetadot_dmd = X_dmd[1, :].real

#         # print(theta_dmd)
#         # print(thetadot_dmd)
        
#         return theta_dmd, thetadot_dmd

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

def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

Num_of_pendulums = 1
total_time=5
dt=0.01
steps = int(total_time/dt)

Time = np.zeros((Num_of_pendulums, steps + 1))
context = 2
start_index = 5

data_path = f'inference_run/onepend_mse_sameinitcond6_125000/results_maxcontext{start_index}.pkl'


# X0, masses, lengths, theta_result_model, thetadot_result_model = load_data(data_path) ## one pend test model
# masses, lengths, theta_result_model, thetadot_result_model = load_data(data_path) ## one pend test model

# masses = [generate_dataset.sample_mass_uniform() for _ in range(Num_of_pendulums)] ## multipend no test model
# lengths = [generate_dataset.sample_length_uniform() for _ in range(Num_of_pendulums)] ## multipend no test model

# theta_model_context = theta_result_model[context-1]  ## one pend test model
# thetadot_model_context = thetadot_result_model[context-1] ## one pend test model


thetas_rk4 = np.zeros((Num_of_pendulums, steps + 1))
thetadots_rk4 = np.zeros((Num_of_pendulums, steps + 1))
thetas_dmd = np.zeros((Num_of_pendulums, steps + 1))
thetadots_dmd = np.zeros((Num_of_pendulums, steps + 1))

i = 0
X0 = [np.pi, 3]
masses = [2.2]
lengths = [1.5]



for mass, length in tqdm(zip(masses, lengths), desc="MultiPendulum", total=len(masses), leave=False):
    # X0 = generate_random_X0()
    print(X0)
    # X0 = [3*np.pi/4,10.0]
    # X0 = [-0.0811,  1.3219]
    # X0 = [-0.08689436, 1.321947] ##### 2/18/2025 (ebonye): same init cond for training
    # X0 = [0.46671834184573213, -2.639731918107688]
    # X0 = [0.46671834184573213, -2.639731918107688]
    # X0 = [np.pi/2,3]
    T1, theta_rk4, thetadot_rk4, control_values_rk4, K_values = workCon.checking(
                X0, total_time, method='rk4', dt=dt, mass = mass, length = length
            )
    xs_dataset = np.column_stack((theta_rk4[start_index-context:start_index+1], thetadot_rk4[start_index-context:start_index+1]))
    xs_dataset = torch.tensor(xs_dataset).float().cuda()


    theta_model_final = np.concatenate((theta_rk4[:steps+1-len(theta_model_context)], theta_model_context)) ## one pend test model
    thetadot_model_final = np.concatenate((thetadot_rk4[:steps+1-len(thetadot_model_context)], thetadot_model_context)) ## one pend test model
    

    dmdinit = xs_dataset[0].cpu().detach().numpy()

    theta_dmd, thetadot_dmd = dynamic_mode_decomposition(xs_dataset, dmdinit, total_time, dt, context) ## one pend test model
    thetas_dmd[i] = theta_dmd ## one pend test model
    thetadots_dmd[i] = thetadot_dmd ## one pend test model
    thetas_rk4[i] = theta_rk4
    thetadots_rk4[i] = thetadot_rk4
    Time[i] = T1
    i += 1



# #### Plot time series plot one pendulum model included
# plt.figure()
# plt.plot(Time[0], thetas_rk4[0], label=f"theta rk4")
# plt.plot(Time[0], thetadots_rk4[0], label=f"thetadot rk4")
# if context == start_index:
#     plt.plot(Time[0, start_index - context:], thetas_dmd[0][:], label=f"theta dmd")
#     plt.plot(Time[0, start_index - context:], thetadots_dmd[0][:], label=f"thetadot dmd")
# else:
#     plt.plot(Time[0, start_index - context:], thetas_dmd[0][:-(start_index-context)], label=f"theta dmd")
#     plt.plot(Time[0, start_index - context:], thetadots_dmd[0][:-(start_index-context)], label=f"thetadot dmd")
# plt.plot(Time[0], theta_model_final, label=f"theta model")
# plt.plot(Time[0], thetadot_model_final, label=f"thetadot model")
# plt.plot(Time[0, start_index - context], theta_model_context[0], 'bo', label='Context examples given')
# plt.plot(Time[0, start_index - context], thetadot_model_context[0], 'bo', label='Context examples given')
# plt.plot(Time[0, start_index-1], thetas_rk4[0, start_index-1], 'ro', label='ICL begins')
# plt.plot(Time[0, start_index-1], thetadots_rk4[0, start_index-1], 'ro', label='ICL begins')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.title(f'RK4 vs DMD: Context length {context}')
# plt.legend()
# plt.savefig('rk4_vs_dmd.png')

# plt.figure()
# plt.plot(thetas_rk4[0], thetadots_rk4[0], marker='x', color='black', label='RK4')
# plt.plot(thetas_dmd[0], thetadots_dmd[0], marker='^', label='DMD')
# plt.plot(theta_model_final, thetadot_model_final, marker='s', alpha=0.2, label='Model')
# plt.plot(theta_model_context[0], thetadot_model_context[0], 'bo', label='Context examples given')
# plt.plot(thetas_rk4[0, start_index-1], thetadots_rk4[0, start_index-1], 'ro', label='ICL begins')
# plt.xlabel('Theta')
# plt.ylabel('ThetaDot')
# plt.title(f'Phase Plot: Context length {context}')
# plt.legend()
# plt.savefig('phase_plot.png')



# #### Plot Multiple Pendulums no model included
# plt.figure()
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
# # for i in range(Num_of_pendulums):
# #     plt.plot(Time[i], thetas_rk4[i], label=f"Pendulum {i} Theta rk4", color=colors[i])
# #     plt.plot(Time[i], thetadots_rk4[i], label=f"Pendulum {i} Thetadot rk4", color=colors[i])
# #     # plt.plot(Time[i], thetas_dmd[i], label=f"Pendulum {i} Theta dmd", color=colors[i], linestyle='--')
# #     # plt.plot(Time[i], thetadots_dmd[i], label=f"Pendulum {i} Thetadot dmd", color=colors[i], linestyle='--')

# #     plt.xlabel('Time')
# #     plt.ylabel('Values')
# #     plt.title('Time Series')

# #     # plt.legend()

# fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# for i in range(Num_of_pendulums):
#     ax[0].plot(Time[i], thetas_rk4[i], label=f"Pendulum {i} Theta rk4", color=colors[i])
#     ax[1].plot(Time[i], thetadots_rk4[i], label=f"Pendulum {i} Thetadot rk4", color=colors[i])
#     # ax[0].plot(Time[i], thetas_dmd[i], label=f"Pendulum {i} Theta dmd", color=colors[i], linestyle='--')
#     # ax[1].plot(Time[i], thetadots_dmd[i], label=f"Pendulum {i} Thetadot dmd", color=colors[i], linestyle='--')

#     ax[0].set_xlabel('Time')
#     ax[0].set_ylabel('Values')
#     ax[0].set_title('Time Series Theta')

#     ax[1].set_xlabel('Time')
#     ax[1].set_ylabel('Values')
#     ax[1].set_title('Time Series ThetaDot')

#     # ax.legend()


# plt.savefig('multipendulum_Timeseries.png')

# plt.figure()
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
# for i in range(Num_of_pendulums):
#     plt.plot(thetas_rk4[i], thetadots_rk4[i], label=f"Pendulum {i} RK4", color=colors[i], marker='x')
#     # plt.plot(thetas_dmd[i], thetadots_dmd[i], label=f"Pendulum {i} DMD", color=colors[i], linestyle='--')
#     if i == 0:
#         print(thetas_rk4[i][-1])
#         print(thetadots_rk4[i][-1])
#     plt.xlabel('Theta')
#     plt.ylabel('ThetaDot')
#     plt.title('Phase Plot')

#     # plt.legend()

# plt.savefig('multipendulum_phaseplot.png')



        