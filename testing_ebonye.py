import torch
import numpy as np

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


datax = torch.tensor(np.random.rand(1, 100, 2))
datay = torch.tensor(np.random.rand(1, 100, 1))

window_size = 10
stride = 1

x_windows, y_windows = window_dataset(datax, datay, window_size, stride)
print(x_windows.shape)
print(y_windows.shape)
