"""
Helper Library for Machine Learning and Data Manipulation
----------------------------------------------------------

Author: Esmail Jamshidiasl (NeuralNinjaEsmail)
Co-author: Microsoft Copilot (AI Assistant)
Version: 1.0.0
Date: 14 March 2025

License:
    MIT License
"""



import numpy as np
import matplotlib.pyplot as plt


def number_to_binary(number, num_bits=16):
    """
    Converts an integer to its binary representation as a NumPy array.

    Parameters:
        number (int): The integer number to be converted to binary.
        num_bits (int, optional): The number of bits in the binary representation.
                                  Defaults to 16.

    Returns:
        numpy.ndarray: A NumPy array containing the binary digits (0s and 1s) of the number.
                       The most significant bit (MSB) is at index 0.

    Example:
        >>> number_to_binary(5)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])
        >>> number_to_binary(5, num_bits=8)
        array([0, 0, 0, 0, 0, 1, 0, 1])
    """

    return np.array([int(bit) for bit in format(number, f'0{num_bits}b')])


def binary_to_number(binary_array):
    """
    Converts a NumPy array representing a binary number to its decimal equivalent.

    Parameters:
        binary_array (numpy.ndarray): A NumPy array containing binary digits (0s and 1s).
                                      The most significant bit (MSB) should be at index 0.

    Returns:
        int: The decimal (integer) representation of the binary number.

    Example:
        >>> binary_to_number(np.array([1, 0, 1, 1]))
        11
    """

    binary_string = ''.join(map(str, binary_array))

    return int(binary_string, 2)


def calc_accuracy(outputs, labels):
    """
    Calculates the accuracy of predictions compared to the true labels.
    and returns it as a percentage.

    Parameters:
        outputs (numpy.ndarray or list): The predicted outputs.
        labels (numpy.ndarray or list): The true labels.

    Returns:
        float: The accuracy as a percentage between 0 and 100, formatted to two decimal places.

    Example:
        >>> outputs = np.array([0, 1, 1, 0, 1])
        >>> labels = np.array([0, 1, 0, 0, 1])
        >>> calc_accuracy(outputs, labels)
        80.00
    """

    outputs = np.array(outputs)
    labels = np.array(labels)

    correct_predictions = np.sum(outputs == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    return round(accuracy, 2)


def normalize_array(arr):
    """
    Normalizes a NumPy array to the range 0-1.

    Parameters:
        arr (numpy.ndarray): The input array to normalize.

    Returns:
        numpy.ndarray: The normalized array with values scaled to the range 0-1.

    Example:
        >>> arr = np.array([10, 20, 30, 40, 50])
        >>> normalize_array(arr)
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val - min_val == 0:
        return np.zeros_like(arr, dtype=float)

    return (arr - min_val) / (max_val - min_val)


def print_train_info(current_epoch, total_epochs, loss):
    """
    Prints the current epoch, total epochs, and loss in a formatted and readable way.

    Parameters:
        current_epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        loss (float): The current loss value.

    Example:
        >>> print_train_info(2, 10, 0.3456)
        Epoch: [2 / 10], Loss: 0.3456
    """

    print(f"Epoch: [{current_epoch} / {total_epochs}], Loss: {loss:.4f}")


def split_array(array, train_ratio=0.8):
    """
    Splits an array into two parts: training and validation.

    Parameters:
        array (numpy.ndarray): The array to split.
        train_ratio (float): Ratio of the data to use for training. Defaults to 0.8.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Training and validation arrays.

    Example:
        >>> array = np.array([1, 2, 3, 4, 5])
        >>> train, val = split_array(array, train_ratio=0.6)
        >>> train, val
        (array([1, 2, 3]), array([4, 5]))
    """

    train_size = int(len(array) * train_ratio)

    return array[:train_size], array[train_size:]


def plot_losses(epoch_losses, every_x_epoch=1):
    """
    Plots only the loss values for every_x_epoch.

    Parameters:
        epoch_losses (list or numpy.ndarray): A list or NumPy array containing loss values for each epoch.
        every_x_epoch (int): The modulus condition to filter which epochs to plot.

    Example:
        >>> losses = [0.8, 0.6, 0.5, 0.3, 0.2, 0.1]
        >>> plot_losses(losses, 2)  # Plots losses for epochs 2, 4, 6
    """

    epochs = np.arange(1, len(epoch_losses) + 1)

    filtered_epochs = epochs[epochs % every_x_epoch == 0]
    filtered_losses = np.array(epoch_losses)[epochs % every_x_epoch == 0]

    plt.figure(figsize=(8, 6))
    plt.plot(filtered_epochs, filtered_losses,
             marker='o', linestyle='-', color='g')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss for Epochs (filtered every {every_x_epoch} epoch)")
    plt.grid(True)
    plt.legend()
    plt.show()


def add_custom_features(array):
    """
    Expands the input array by adding x^2, x^3, sign(x), e^x, cos(x), and a sign indicator
    as additional features.

    Parameters:
        array (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: A new array with the original values and the additional features.

    Example:
        >>> arr = np.array([-2, -1, 0, 1, 2])
        >>> add_custom_features(arr)
        array([[-2.        ,  4.        , -8.        , -1.        ,  0.13533528, -0.41614684, -1.        ],
               [-1.        ,  1.        , -1.        , -1.        ,  0.36787944,  0.54030231, -1.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,  1.        ,  0.        ],
               [ 1.        ,  1.        ,  1.        ,  1.        ,  2.71828183,  0.54030231,  1.        ],
               [ 2.        ,  4.        ,  8.        ,  1.        ,  7.3890561 , -0.41614684,  1.        ]])
    """

    array = np.array(array)

    squared = array ** 2
    cubed = array ** 3
    sign = np.sign(array)
    exponential = np.exp(array)
    cosine = np.cos(array)
    sign_indicator = np.where(array > 0, 1, np.where(array < 0, -1, 0))

    expanded_array = np.column_stack(
        (array, squared, cubed, sign, exponential, cosine, sign_indicator))

    return expanded_array


def standardize_array(array):
    """Standardizes the input array to have mean=0 and std=1."""

    return (array - np.mean(array)) / np.std(array)


def min_max_scale(array, a=0, b=1):
    """Scales the input array to the range [a, b]."""

    min_val, max_val = np.min(array), np.max(array)

    return a + (array - min_val) * (b - a) / (max_val - min_val)
