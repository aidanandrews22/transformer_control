import os
import pickle
import numpy as np

# pickle_dir = "dataset_multipendulum_gaussian/saved_pickles"
# pickle_dir = "dataset_pendulum/picklefolder_uniform_sameinitcond"
pickle_dir = "dataset_pendulum/picklefolder"
# pickle_dir = "dataset_pendulum/picklefolder_toobig_gaussian"

file_number = 50 #205000 

pickle_file = f"multipendulum_{file_number}.pkl"

file_path = os.path.join(pickle_dir, pickle_file)

if not os.path.exists(file_path):
    print(f"File '{pickle_file}' does not exist in '{pickle_dir}'.")
else:
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"Contents of {pickle_file}:")
        print(np.shape(data[0]))
        # print(np.shape(data[1]))

        # print(data[0])
        # datanp = data
        # print(np.shape(datanp))
    except Exception as e:
        print(f"An error occurred while reading {pickle_file}: {e}")
