import math
import torch
import numpy as np
from workCon import checking  

class DataSampler:
    # def __init__(self, n_dims, init_conditions):
    def __init__(self, n_dims):
        self.n_dims = n_dims
        # self.X0 = init_conditions
        

    def sample_xs(self):
        raise NotImplementedError

def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError

def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class PendulumSampler(DataSampler):
    # def __init__(self, n_dims, init_conditions):
    #     super().__init__(n_dims, init_conditions)

    def __init__(self, n_dims):
        super().__init__(n_dims)    
        # self.theta_limit = np.pi  
        # self.thetadot_limit = 10.0  
        self.theta_limit = np.pi/4
        self.thetadot_limit = 6.0 ###### 2/8/2025 (ebonye): changing bounds to make it easier for the model to learn

    def sample_val_initial_conditions(self):
        """
        Samples initial conditions for validation set, ensuring initial theta and thetadot don't overlap with training set

        Returns:
            list: A list containing:
                - theta_init (float): The initial angular position (theta) in radians.
                - thetadot_init (float): The initial angular velocity (thetadot).
        """
        epsilon = 1e-6  
        theta_ranges = [(-3 * np.pi / 2, -np.pi - epsilon), (np.pi + epsilon, 3 * np.pi / 2)]
        theta_choice = np.random.choice([0, 1])
        theta_init = np.random.uniform(*theta_ranges[theta_choice])
        thetadot_ranges = [(-20.0, -11.0), (11.0, 20.0)]
        thetadot_choice = np.random.choice([0, 1])
        thetadot_init = np.random.uniform(*thetadot_ranges[thetadot_choice])
        return [theta_init, thetadot_init]

    def sample_initial_conditions(self):
        """
        Samples general initial conditions for the pendulum system within the limits.

        Returns:
            list: A list containing:
                - theta_init (float): The initial angular position (theta) in radians, sampled uniformly from [-pi, pi].
                - thetadot_init (float): The initial angular velocity (thetadot), sampled uniformly from [-10, 10].
        """
        # theta_init = np.random.uniform(-self.theta_limit, self.theta_limit)
        # theta_init = np.random.uniform(np.pi/5, np.pi/2) ###### 2/8/2025 (ebonye): changing bounds to make it easier for the model to learn
        # theta_init = np.random.uniform(np.pi/5, 7*np.pi/6) ###### 3/5/2025 (ebonye): making more difficult
        # theta_init = np.random.uniform(np.pi-(np.pi/2), np.pi+(np.pi/2)) ###### 3/9/2025 (ebonye): upswing initial conditions
        # thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit)
        theta_init = np.random.uniform(np.pi - (3*np.pi/4), np.pi + (3*np.pi/4)) ###### 3/9/2025 (ebonye): upswing initial conditions
        thetadot_init = np.random.uniform(-self.thetadot_limit, self.thetadot_limit)
        return [theta_init, thetadot_init]

    def calculate_lyapunov_derivative(self, theta, thetadot, u, m=1, l=1, b=0.5, g=9.81):
        """
        Calculates the time derivative of the Lyapunov function for a pendulum system, 
        given its current state and control input. The Lyapunov derivative provides insights 
        into the stability of the system and whether the control law drives the system toward 
        the desired state.

        Args:
            theta (float or torch.Tensor): The angular position (in radians) of the pendulum.
            thetadot (float or torch.Tensor): The angular velocity of the pendulum.
            u (float or torch.Tensor): The control input applied to the system.
            m (float, optional): The mass of the pendulum. Defaults to 1.
            l (float, optional): The length of the pendulum. Defaults to 1.
            b (float, optional): The damping coefficient. Defaults to 0.5.
            g (float, optional): The acceleration due to gravity. Defaults to 9.81.

        Returns:
            torch.Tensor: The time derivative of the Lyapunov function, indicating the rate of change of the system's energy under the given state and control input. (All values should be negative)
        """
        theta = torch.tensor(theta, dtype=torch.float32) if not isinstance(theta, torch.Tensor) else theta
        thetadot = torch.tensor(thetadot, dtype=torch.float32) if not isinstance(thetadot, torch.Tensor) else thetadot
        u = torch.tensor(u, dtype=torch.float32) if not isinstance(u, torch.Tensor) else u
        dV_dtheta = m * g * l * torch.sin(theta)
        dV_dthetadot = m * l**2 * thetadot
        ddot_theta = (-b * thetadot + m * g * l * torch.sin(theta) + u) / (m * l**2)

        dV_dt = dV_dtheta * thetadot + dV_dthetadot * ddot_theta
        return dV_dt
    
    # def generate_xs_dataset(self, n_points, b_size, val = "no", mass=1,length=1):
    def generate_xs_dataset(self, n_points, mass=1, length=1):
        """
        Generates datasets for training or evaluation by simulating the states and control values of pendulum system.

        Args:
            n_points (int): The total number of points to generate for the simulation. This value determines the time steps for each trajectory.
            # b_size (int): The batch size
            val (str, optional): A flag to indicate whether this is a validation dataset or not. Defaults to "no".
            mass (float, optional): The mass of the system being simulated. Defaults to 1.
            length (float, optional): The length of the pendulum or system being simulated. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - T_batches (torch.Tensor): A tensor of time steps for each trajectory in the batch.
                - xs_datasets (torch.Tensor): A tensor of state datasets (theta and thetadot) for each trajectory.
                - control_values_batches (torch.Tensor): A tensor of control values for each trajectory.
                - k_values (np.ndarray): The gain matrix of the pendulum simulation.
        """
        T_batches = []
        xs_datasets = []
        control_values_batches = []
        n_stop = n_points
        n_points = n_points/100
        # if self.X0 is None:
        #     self.X0 = self.sample_initial_conditions() #2/8/2025 (ebonye): each batch has the same initial conditions


        # for _ in range(b_size):     
        # ebonye 2/28/2025: removing loop to mix mass/length in a batch right before training       
        X0 = self.sample_initial_conditions() #get starting initial values of theta and thetadot
        T, theta, thetadot, control_values, k_values = checking(X0, n_points, method='rk4', dt=0.01,mass = mass,length = length) # where simulation happens
        # T, theta, thetadot, control_values, k_values = checking(self.X0, n_points, method='rk4', dt=0.01,mass = mass,length = length)
        T = T[:n_stop]
        theta = theta[:n_stop]
        thetadot = thetadot[:n_stop]
        control_values = control_values[:n_stop]
        
        xs_dataset = np.column_stack((theta, thetadot))

        T_batches.append(T) 
        xs_datasets.append(xs_dataset)
        control_values_batches.append(control_values)


        T_batches = np.array(T_batches)
        xs_datasets = np.array(xs_datasets)
        control_values_batches = np.array(control_values_batches)
        T_batches = torch.tensor(T_batches).float()
        xs_datasets = torch.tensor(xs_datasets).float()
        control_values_batches = torch.tensor(control_values_batches).float()

        if torch.cuda.is_available():
            T_batches = T_batches.cuda()
            xs_datasets = xs_datasets.cuda()
            control_values_batches = control_values_batches.cuda()

        return T_batches, xs_datasets, control_values_batches, k_values



