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
import plotly.graph_objects as go


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

        # Atilde = Y @ V.T @ np.linalg.inv(np.diag(S)) @ U.T
        # eigvals, eigvecs = np.linalg.eig(Atilde)
        
        Phi = Y @ V.T @ np.linalg.inv(np.diag(S)) @ eigvecs
        # Phi = eigvecs
        # b = np.linalg.pinv(Phi) @ X[:, 0]
        b = np.linalg.pinv(Phi) @ X0

        # print(f'Eigvals: {eigvals}')
        Omega = np.log(eigvals) / dt
        # Omega = np.clip(Omega, -200, 10)
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

model_run_id = "9bb50653-5ed4-49c1-8dae-a876b2677236" #"3c33d621-e18a-4c4b-9844-54915b1de7b1"
data_path = f'inference_run/mse_control_39100_{model_run_id}/results_maxcontext20_numpends1_indistr.pkl'
X0s_stored, masses, lengths, phase_data, data_and_controls, pends = load_data(data_path)
# print(f'masses: {masses}')
# print(f'lengths: {lengths}')
'''
X0s_stored: list of initial conditions for each pendulum
masses: list of masses for each pendulum
lengths: list of lengths for each pendulum
phase_data: dictionary of phase data where keys are context length and values are list of phase data for each pendulum
data_and_controls: list of rk4 data and controls for each pendulum
pends: pendulum indices for pickle files
'''
###################################################################################################
##### Plotting Phase Space
total_time = 5
dt = 0.01
pendulum_index = 0


T = np.arange(0, total_time, dt)
ground_truth_data_controls = data_and_controls[pendulum_index]
theta_rk4 = (np.squeeze(ground_truth_data_controls[0]).cpu().detach().numpy())[:, 0]
thetadot_rk4 = (np.squeeze(ground_truth_data_controls[0]).cpu().detach().numpy())[:, 1]


plt.figure(figsize=(10, 6))
plt.plot(theta_rk4, thetadot_rk4, label='Ground Truth', color='black', marker = 'X')

for context in phase_data.keys():
    plt.plot(phase_data[context][pendulum_index][:,0], phase_data[context][pendulum_index][:,1], label=f'Context: {context}', marker = 'o')
    # theta_dmd, thetadot_dmd = dynamic_mode_decomposition(phase_data[context][pendulum_index], X0s_stored[pendulum_index], total_time, dt, context)
    # plt.plot(theta_dmd, thetadot_dmd, label=f'DMD: Context {context}', marker = 'o')



plt.xlabel('Theta')
plt.ylabel('Theta_dot')
plt.title('Phase Space')
plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.savefig('multipendulum_phaseplot.png', bbox_inches='tight', dpi=300)


fig = go.Figure()
fig.add_trace(go.Scatter(x=theta_rk4, y=thetadot_rk4, name='Ground Truth', mode='markers', marker=dict(color='black')))
for context in phase_data.keys():
    fig.add_trace(go.Scatter(x=phase_data[context][pendulum_index][:,0], y=phase_data[context][pendulum_index][:,1], mode='markers', name=f'Context: {context}'))
    if context != 1:
        theta_dmd, thetadot_dmd = dynamic_mode_decomposition(phase_data[context][pendulum_index], X0s_stored[pendulum_index], total_time, dt, context)
        # fig.add_trace(go.Scatter(x=theta_dmd, y=thetadot_dmd, mode='markers', name=f'DMD: Context {context}'))

fig.update_layout(title='Phase Space', xaxis_title='Theta', yaxis_title='Theta_dot')
# fig.write_image('multipendulum_phaseplot_plotly.png')
# fig.write_html('multipendulum_phaseplot_plotly.html')
fig.show()

###################################################################################################3
##### plotting mse vs context length + error bars for all pendulums
mse_values = {context: [] for context in phase_data.keys()} # dictionary to store mse values for each context length}
mse_values_dmd = {context: [] for context in phase_data.keys()} # dictionary to store mse values for each context length}
# trajectories = 
# mse_values_dmd[1] = [0] * len(pends)
del mse_values_dmd[1]
for pendulum_index in range(len(pends)):
    ground_truth_data_controls = data_and_controls[pendulum_index]
    theta_rk4 = (np.squeeze(ground_truth_data_controls[0]).cpu().detach().numpy())[:, 0]
    thetadot_rk4 = (np.squeeze(ground_truth_data_controls[0]).cpu().detach().numpy())[:, 1]
    for context in phase_data.keys():
        theta_model = phase_data[context][pendulum_index][:,0]
        thetadot_model = phase_data[context][pendulum_index][:,1]
        mse_val = mse(theta_model, thetadot_model, theta_rk4, thetadot_rk4)
        mse_values[context].append(mse_val)
        # if context != 1:
            # print(f'Pendulum: {pendulum_index}, Context: {context}')
            # theta_dmd, thetadot_dmd = dynamic_mode_decomposition(phase_data[context][pendulum_index], X0s_stored[pendulum_index], total_time, dt, context)
            # mse_val_dmd = mse(theta_dmd, thetadot_dmd, theta_rk4, thetadot_rk4)
            # mse_values_dmd[context].append(mse_val_dmd)

mean_mse_values = {context: np.mean(mse_values[context]) for context in mse_values.keys()}
std_mse_values = {context: np.std(mse_values[context]) for context in mse_values.keys()}
median_mse_values = {context: np.median(mse_values[context]) for context in mse_values.keys()}
lower_bound = {context: np.percentile(mse_values[context], 25) for context in mse_values.keys()}
upper_bound = {context: np.percentile(mse_values[context], 75) for context in mse_values.keys()}
plt.figure(figsize=(10, 6))

lower_bound_std = np.array([std if mean - std > 0 else mean for mean, std in zip(mean_mse_values.values(), std_mse_values.values())])
upper_bound_std = np.array(list(std_mse_values.values()))
# plt.errorbar(mean_mse_values.keys(), mean_mse_values.values(), yerr=[lower_bound_std, upper_bound_std], fmt='o-', capsize=5, color='black')
# plt.errorbar(mean_mse_values.keys(), mean_mse_values.values(), yerr=std_mse_values.values(), fmt='o-', capsize=5, color='black')
plt.errorbar(median_mse_values.keys(), median_mse_values.values(), yerr=[np.array(list(median_mse_values.values())) - np.array(list(lower_bound.values())), np.array(list(upper_bound.values())) - np.array(list(median_mse_values.values()))], fmt='o-', capsize=5, color='blue')
# plt.xticks(list(median_mse_values.keys()))

# mean_mse_values_dmd = {context: np.mean(mse_values_dmd[context]) for context in mse_values_dmd.keys()}
# std_mse_values_dmd = {context: np.std(mse_values_dmd[context]) for context in mse_values_dmd.keys()}
# lower_bound_std_dmd = np.array([std if mean - std > 0 else mean for mean, std in zip(mean_mse_values_dmd.values(), std_mse_values_dmd.values())])
# upper_bound_std_dmd = np.array(list(std_mse_values_dmd.values()))
# plt.errorbar(mean_mse_values_dmd.keys(), mean_mse_values_dmd.values(), yerr=std_mse_values_dmd.values(), fmt='o-', capsize=5, color='red')


# median_mse_values_dmd = {context: np.median(mse_values_dmd[context]) for context in mse_values_dmd.keys()}
# lower_bound_dmd = {context: np.percentile(mse_values_dmd[context], 25) for context in mse_values_dmd.keys()}
# upper_bound_dmd = {context: np.percentile(mse_values_dmd[context], 75) for context in mse_values_dmd.keys()}
# plt.errorbar(median_mse_values_dmd.keys(), median_mse_values_dmd.values(), yerr=[np.array(list(median_mse_values_dmd.values())) - np.array(list(lower_bound_dmd.values())), np.array(list(upper_bound_dmd.values())) - np.array(list(median_mse_values_dmd.values()))], fmt='o-', capsize=5, color='red')
plt.grid()
plt.yscale('log')
# plt.ylim(1e-8, 1e-1)
plt.grid()
plt.xticks(list(mean_mse_values.keys())[::4])
plt.xlabel('Context Length')
plt.ylabel('MSE')
plt.title('MSE vs Context Length')
plt.savefig('multipendulum_mse_vs_context.png', bbox_inches='tight', dpi=300)
plt.show()

#############################################################################################
##### Plotting Checkpoint versus MSE
# all_results = load_data('all_results_5pend_indistr_3c33d621-e18a-4c4b-9844-54915b1de7b1.pkl')
# all_results = load_data('all_results_5pend_ood_3c33d621-e18a-4c4b-9844-54915b1de7b1.pkl')
# all_results = load_data('all_results_5pend_indistr_9bb50653-5ed4-49c1-8dae-a876b2677236.pkl')
all_results = load_data('all_results_5pend_ood_9bb50653-5ed4-49c1-8dae-a876b2677236.pkl')
mse_results = all_results['mse_results']
mse_controls_results = all_results['mse_controls_results']
checkpoints = list(mse_results.keys())
mse_values = list(mse_results.values())
mse_controls_values = list(mse_controls_results.values())
median_mse_values = {step: np.median(mse_values) for step, mse_values in mse_results.items()}
min_mse_values = {step: np.min(mse_values) for step, mse_values in mse_results.items()}
max_mse_values = {step: np.max(mse_values) for step, mse_values in mse_results.items()}

median_mse_controls_values = {step: np.median(mse_values) for step, mse_values in mse_controls_results.items()}
min_mse_controls_values = {step: np.min(mse_values) for step, mse_values in mse_controls_results.items()}
max_mse_controls_values = {step: np.max(mse_values) for step, mse_values in mse_controls_results.items()}
plt.figure(figsize=(15, 6))
# plt.plot(checkpoints, mse_values, marker='o', color='black')
plt.plot(checkpoints, list(median_mse_values.values()), marker='o', color='black', label='MSE state')
# plt.fill_between(checkpoints, list(min_mse_values.values()), list(max_mse_values.values()), color='gray', alpha=0.5)
# plt.plot(checkpoints, list(median_mse_controls_values.values()), marker='o', color='red', label='MSE control')
# plt.fill_between(checkpoints, list(min_mse_controls_values.values()), list(max_mse_controls_values.values()), color='pink', alpha=0.5)
plt.legend()
plt.yscale('log')
plt.xlabel('Checkpoint')
plt.ylabel('MSE (log)')
# plt.xticks(np.arange(0, checkpoints[-1], 10000))
plt.title('MSE vs Checkpoint')
plt.savefig('mse_vs_checkpoint.png', bbox_inches='tight', dpi=300)
plt.show()
# print(checkpoints)
# print(np.min(list(median_mse_values.values())))

###################################################################################################
##### Plotting Training Points
pends = np.arange(0, 50)
# plt.figure(figsize=(10, 6))
# for i in range(len(pends)):
#     print(f'Pendulum: {i}')
#     datapath = f'dataset_pendulum/picklefolder/multipendulum_{i}.pkl'
#     data_controls = load_data(datapath)
#     theta_rk4 = (np.squeeze(data_controls[0]).cpu().detach().numpy())[:, 0]
#     thetadot_rk4 = (np.squeeze(data_controls[0]).cpu().detach().numpy())[:, 1]
#     control_values_rk4 = (np.squeeze(data_controls[1]).cpu().detach().numpy())
#     plt.plot(theta_rk4, thetadot_rk4, label=f'Pendulum: {i}', marker = 'o')

# plt.xlabel('Theta')
# plt.ylabel('Theta_dot')
# plt.title('Training Points')
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
# plt.savefig('multipendulum_trainingpoints.png', bbox_inches='tight', dpi=300)
# # plt.show()

            




