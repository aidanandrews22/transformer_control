import numpy as np
from scipy.linalg import solve_continuous_are
from numba import njit

def batch_lqr_gains(masses, lengths, Q, R):
    """
    Computes LQR gains for multiple systems in a batch.
    """
    n = len(masses)
    K_all = np.zeros((n, 1, 2))  # shape (systems, 1, 2)
    for i in range(n):
        m, l = masses[i], lengths[i]
        A = np.array([[0, 1],
                      [9.81 / l, -0.5 / (m * l ** 2)]])
        B = np.array([[0],
                      [1 / (m * l ** 2)]])
        X = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ (B.T @ X)
        K_all[i] = K
    return K_all  # shape (n, 1, 2)

@njit
# def simulate_batch(X0_list, K_all, dt, n_steps):
def simulate_batch(X0_list, K_all, masses, lengths, dt, n_steps):
    """
    Vectorized RK4 simulation using LQR feedback for multiple systems.
    """
    num_systems = X0_list.shape[0]
    theta_all = np.zeros((num_systems, n_steps))
    thetadot_all = np.zeros((num_systems, n_steps))
    tau_all = np.zeros((num_systems, n_steps))

    for i in range(num_systems):
        x = X0_list[i]
        K = K_all[i]
        theta_all[i, 0] = x[0]
        thetadot_all[i, 0] = x[1]
        mass, length = masses[i], lengths[i]

        for t in range(n_steps - 1):
            u = -K @ (x - np.array([0.0, 0.0]))
            tau_all[i, t] = u[0]
            # x = rk4_step(x, u[0], dt)
            x = rk4_step(x, u[0], dt, mass, length)
            theta_all[i, t + 1] = x[0]
            thetadot_all[i, t + 1] = x[1]

    return theta_all, thetadot_all, tau_all

@njit
# def rk4_step(x, u, dt):
def rk4_step(x, u, dt, m=1, l=1):
    """
    RK4 integrator for a single system step.
    """
    def f(x, u):
        # m, l, b, g = 1.0, 1.0, 0.5, 9.81
        b, g = 0.5, 9.81
        theta, thetadot = x
        dtheta = thetadot
        dthetadot = (-b * thetadot + m * g * l * np.sin(theta) + u) / (m * l ** 2)
        return np.array([dtheta, dthetadot])

    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_all_simulations(X0_list, total_time, dt, masses, lengths):
    n_steps = int(total_time / dt) + 1
    Q = np.eye(2)
    R = np.array([[1]])
    K_all = batch_lqr_gains(masses, lengths, Q, R)
    # theta, thetadot, tau = simulate_batch(X0_list, K_all, dt, n_steps)
    theta, thetadot, tau = simulate_batch(X0_list, K_all, masses, lengths, dt, n_steps) 
    T = np.linspace(0, total_time, n_steps)
    return T, theta, thetadot, tau


def checking(X0, total_time, masses, lengths, method='rk4', dt=0.01):
    """
    Checks the simulation of inverted pendulum systems using either RK4 or Euler methods.
    
    Args:
        X0 (list or np.ndarray): Initial state of the pendulum [theta, thetadot].
        total_time (float): Total duration of the simulation in seconds.
        masses (np.ndarray): Array of mass values for each system.
        lengths (np.ndarray): Array of length values for each system.
        method (str, optional): Method to use for simulation ('rk4' or 'euler'). Defaults to 'rk4'.
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
    
    Returns:
        T, theta_all, thetadot_all, tau_all: Simulation results for each system.
    """
    if method == 'rk4':
        # T, theta_all, thetadot_all, tau_all = vectorized_simulation(X0, total_time, dt, masses, lengths)
        T, theta_all, thetadot_all, tau_all = run_all_simulations(X0, total_time, dt, masses, lengths)
    else:
        raise ValueError("Invalid method. Choose 'rk4'.")
    
    return T, theta_all, thetadot_all, tau_all

##################################################################


## Unbatched and unvectorized
# import numpy as np

# import numpy as np
# from scipy.linalg import solve_continuous_are

# def single_step_inverted_pendulum_rk4(X0, u, dt=0.01, mass = 1, length = 1):
#     """
#     Performs a single simulation step for an inverted pendulum system using the 
#     4th-order Runge-Kutta (RK4) method to get next state of theta and thetadot (used for inference).

#     Args:
#         X0 (list or np.ndarray): The current state of the pendulum [theta, thetadot], where:
#             - theta (float): The angular position (in radians).
#             - thetadot (float): The angular velocity.
#         u (float): The control input applied to the pendulum.
#         dt (float, optional): The time step for the simulation. Defaults to 0.01.
#         mass (float, optional): The mass of the pendulum. Defaults to 1.
#         length (float, optional): The length of the pendulum. Defaults to 1.

#     Returns:
#         tuple: A tuple containing:
#             - theta_next (float): The angular position (in radians) after one time step.
#             - thetadot_next (float): The angular velocity after one time step.

#     """
#     params = {'m': mass, 'l': length, 'b': 0.5, 'g': 9.81}
    
#     def f(x, u):
#         m, l, b, g = params["m"], params["l"], params["b"], params["g"]
#         theta = x[0]
#         thetadot = x[1]
#         dtheta = thetadot
#         dthetadot = (-b * thetadot + m * g * l * np.sin(theta) + u) / (m * l**2)
        
#         return np.array([dtheta, dthetadot.item()])

#     x = np.array(X0, dtype=float)
    
#     k1 = f(x, u)
#     k2 = f(x + 0.5 * dt * k1, u)
#     k3 = f(x + 0.5 * dt * k2, u)
#     k4 = f(x + dt * k3, u)
#     x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
#     return x_next[0], x_next[1]



# # def simulate_inverted_pendulum_rk4(X0, total_time, dt=0.01, K=None,  mass = 1, length=1):
# def simulate_inverted_pendulum_rk4(X0, total_time, dt=0.01, mass = 1, length=1):
#     """
#     Simulates the dynamics of an inverted pendulum using the Runge-Kutta 4th-order method (RK4),
#     with an optional state-feedback control law (LQR) applied.

#     Args:
#         X0 (list or np.ndarray): Initial state of the pendulum [theta, thetadot].
#             - theta (float): Initial angular position in radians.
#             - thetadot (float): Initial angular velocity.
#         total_time (float): Total duration of the simulation in seconds.
#         dt (float, optional): Time step for the simulation. Defaults to 0.01.
#         K (np.ndarray, optional): The LQR gain matrix used for state-feedback control. 
#         mass (float, optional): Mass of the pendulum Defaults to 1.
#         length (float, optional): Length of the pendulum. Defaults to 1.

#     Raises:
#         ValueError: If the control gain matrix K is not provided.

#     Returns:
#         tuple: A tuple containing:
#             - T (np.ndarray): Array of time steps during the simulation.
#             - theta (np.ndarray): Array of angular positions over time.
#             - thetadot (np.ndarray): Array of angular velocities over time.
#             - tau (np.ndarray): It's the (u) used for stabilizing.

#     Notes:
#         - The control input u is computed using the state-feedback control law: u = -K(x - x_d).
#         - The desired state x_d is [theta_d, thdot_d], which is set to [0, 0] by default.
#         - The system dynamics are defined as:
#           - velocity: d(theta)/dt = thetadot     
#           - acceleration: d(thetadot)/dt = (-b * thetadot + m * g * l * sin(theta) + u) / (m * l^2)
#         - RK4 is used to find next states.
#     """
#     params = {'m': mass, 'l': length, 'b': 0.5, 'g': 9.81}
    
#     # m = mass
#     # l = length
#     # b = 0.5
#     # g = 9.81

#     # A = np.array([[0, 1],
#     #               [g/l, -b/(m*l**2)]])
#     # B = np.array([[0],
#     #               [1/(m*l**2)]])

    
#     # # Q = np.eye(2)    #  2x2 identity matrix for Q (just keep it constant but can tune)
#     Q = np.diag([1, 1]) 
#     R = np.array([[1]])  # # scalar 1 for R (can change but just keep constant for simplicity)
#     # K = compute_lqr_gain(A, B, Q, R)


#     # K = np.squeeze(K)
 
#     # if K is None:
#         # raise ValueError("no K LQR.")

#     T = np.arange(0, total_time + dt, dt)
#     n_steps = len(T)

#     x = np.array(X0, dtype=float)
#     theta = np.zeros(n_steps)
#     thetadot = np.zeros(n_steps)
#     tau = np.zeros(n_steps)

#     theta[0] = x[0]
#     thetadot[0] = x[1]

#     theta_d = 0.0
#     thdot_d = 0.0

#     def linearized_dynamics(state):
#         m, l, b, g = params["m"], params["l"], params["b"], params["g"]
#         theta, thetadot = state
#         A = np.array([[0, 1],
#                       [g/l * np.cos(theta), -b/(m*l**2)]])
#         B = np.array([[0],
#                       [1/(m*l**2)]])
#         return A, B


#     def f(x, u):
#         m, l, b, g = params["m"], params["l"], params["b"], params["g"]
#         theta = x[0]
#         thetadot = x[1]
#         # print('u: ', u) 
#         # print("theta: ", theta)
#         # print("thetadot: ", thetadot)
#         dtheta = thetadot
#         dthetadot = (-b * thetadot + m * g * l * np.sin(theta) + u) / (m * l**2)
#         # print("dtheta: ", dtheta)
#         # print("dthetadot: ", dthetadot)
#         return np.array([dtheta, dthetadot])

#     for i in range(n_steps - 1):
#         # u = -K @ (x - np.array([theta_d, thdot_d]))
#         # u = 0
#         state = np.array([x[0], x[1]], dtype=float)
#         # print("state: ", state)
#         # print("state type: ", type(state))
#         A, B = linearized_dynamics(state)
#         K = compute_lqr_gain(A, B, Q, R)
#         K = np.squeeze(K)
#         u = -K @ (x - np.array([theta_d, thdot_d]))

#         tau[i] = u

#         # RK4
#         k1 = f(x, u)
#         k2 = f(x + 0.5 * dt * k1, u)
#         k3 = f(x + 0.5 * dt * k2, u)
#         k4 = f(x + dt * k3, u)
#         x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

#         theta[i + 1] = x[0]
#         thetadot[i + 1] = x[1]
#     state = np.array([x[0], x[1]], dtype=float)
#     A, B = linearized_dynamics(state)
#     K = compute_lqr_gain(A, B, Q, R)
#     K = np.squeeze(K)
#     u = -K @ (x - np.array([theta_d, thdot_d]))
#     # u = 0
#     tau[-1] = u

#     return T, theta, thetadot, tau

# def run_lqr_and_simulate(X0, total_time, dt=0.01, mass = 1, length = 1, check = False):
#     """
#     Computes the LQR gain matrix (K) for an inverted pendulum system and simulates its dynamics 
#     using the Runge-Kutta method (RK4). Optionally, the function can return only the LQR gain matrix 
#     without running the simulation (was using it for debugging).

#     Args:
#         X0 (list or np.ndarray): The initial state of the pendulum system [theta, thetadot], where:
#             - theta (float): Initial angular position (in radians).
#             - thetadot (float): Initial angular velocity.
#         total_time (float): Total simulation time (in seconds).
#         dt (float, optional): Time step for the simulation. Defaults to 0.01.
#         mass (float, optional): The mass of the pendulum. Defaults to 1.
#         length (float, optional): The length of the pendulum. Defaults to 1.
#         check (bool, optional): If True, return the LQR gain matrix (K) without running the simulation. Defaults to False.

#     Returns:
#         If `check` is True:
#             np.ndarray: The LQR gain matrix (K), a 1x2 row vector used for the state feedback control.

#         If `check` is False:
#             tuple: A tuple containing:
#                 - T (np.ndarray): Array of time steps during the simulation.
#                 - theta (np.ndarray): Array of angular positions over time.
#                 - thetadot (np.ndarray): Array of angular velocities over time.
#                 - tau (np.ndarray): Array of u's that were used.
#                 - K (np.ndarray): The LQR gain matrix (K).
#     """
#     m = mass
#     l = length
#     b = 0.5
#     g = 9.81

#     A = np.array([[0, 1],
#                   [g/l, -b/(m*l**2)]])
#     B = np.array([[0],
#                   [1/(m*l**2)]])

    
#     # # Q = np.eye(2)    #  2x2 identity matrix for Q (just keep it constant but can tune)
#     # Q = np.diag([1, 1]) 
#     # R = np.array([[1]])  # # scalar 1 for R (can change but just keep constant for simplicity)
#     # #### 2/24/2025 penalize control input ^
    
#     # K = compute_lqr_gain(A, B, Q, R)
#     # if check == True:
#     #     return K

#     # T, theta, thetadot, tau = simulate_inverted_pendulum_rk4(X0, total_time, dt, K=K, mass = mass, length=length)
#     T, theta, thetadot, tau = simulate_inverted_pendulum_rk4(X0, total_time, dt, mass = 0.4, length=0.4)
#     return T, theta, thetadot, tau


# def compute_lqr_gain(A, B, Q, R):
#     """
#     Compute the Linear-Quadratic Regulator (LQR) gain matrix K for the continuous-time system:
#     x_dot = A x + B u, minimizing the cost function:
#         J = ∫ (x^T Q x + u^T R u) dt

#     Args:
#         A (np.ndarray): The state-space matrix representing the system dynamics.
#         B (np.ndarray): The input matrix representing how control inputs affect the system.
#         Q (np.ndarray): The state cost matrix, defining the penalty on the states.
#         R (np.ndarray): The control input cost matrix, defining the penalty on the control inputs.

#     Returns:
#         np.ndarray: The LQR gain matrix K, which determines the optimal control law u = -Kx
#                     to minimize the cost function J.
#     """

#     X = solve_continuous_are(A, B, Q, R)
#     K = np.linalg.inv(R) @ (B.T @ X)
#     return K


# def simulate_inverted_pendulum_euler(X0, total_time, dt=0.01):
#     params = {'m': 1, 'l': 1, 'b': 0.5, 'g': 9.81}
#     K = np.array([23.81842961,  7.10834142])

#     T = np.arange(0, total_time + dt, dt)
#     n_steps = len(T)

#     x = np.array(X0, dtype=float)
#     theta = np.zeros(n_steps)
#     thetadot = np.zeros(n_steps)
#     tau = np.zeros(n_steps)

#     theta[0] = x[0]
#     thetadot[0] = x[1]

#     for i in range(n_steps - 1):
#         t = T[i]
#         theta_d = 0.0
#         thdot_d = 0.0

#         u = -K @ (x - np.array([theta_d, thdot_d]))
#         tau[i] = u

#         theta_dot = x[1]
#         thetadot_dot = (
#             -params["b"] * x[1] + params["m"] * params["g"] * params["l"] * np.sin(x[0]) + u
#         ) / (params["m"] * params["l"] ** 2)

#         x[0] = x[0] + dt * theta_dot
#         x[1] = x[1] + dt * thetadot_dot

#         theta[i + 1] = x[0]
#         thetadot[i + 1] = x[1]

#     u = -K @ (x - np.array([theta_d, thdot_d]))
#     tau[-1] = u

#     return T, theta, thetadot, tau

# def checking(X0, total_time, method='rk4', dt=0.01, mass=1,length=1):
#     if method == 'rk4':
#         T, theta, thetadot, control_values = run_lqr_and_simulate(X0, total_time, dt, mass = mass, length=length)
#     elif method == 'euler':
#         T, theta, thetadot, control_values = simulate_inverted_pendulum_euler(X0, total_time, dt)
#     else:
#         raise ValueError("Invalid method. Choose 'rk4' or 'euler'.")
#     return T, theta, thetadot, control_values