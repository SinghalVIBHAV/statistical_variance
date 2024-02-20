import numpy as np

def compute_variance(partition_A: np.array, partition_b: np.float64) -> float:
    """
    Calculate the variance of a weighted dataset using Schubert & Gertz's method.

    Parameters:
    - partition_A: NumPy array with shape (n, 2), where n is the number of data points.
                   Each row should represent a data point with the first column as weights
                   and the second column as values.
    - partition_b: NumPy array with shape (2,) representing additional data points.

    Returns:
    - The computed variance of the combined dataset.

    Raises:
    - ValueError: If the input is not a NumPy array, not a floating-point or integer array
                  contains NaN values, has less than two dimensions, or doesn't have two columns.
                  Also raised if omega_A is 0 or NaN during the computation.
    """
     
    try:
        if not isinstance(partition_A, np.ndarray) or not isinstance(partition_b, np.ndarray):
            raise ValueError("Both partitions must be NumPy arrays")
        
        if not all(isinstance(val, (np.float64, np.int64, float)) for val in np.concatenate([partition_A.flatten(), partition_b])):
            raise ValueError("All elements of partition_A and partition_b must be floating-point numbers")

        if np.any(np.isnan(partition_A)) or np.any(np.isnan(partition_b)):
            raise ValueError("Input array contains NaN values")

        if partition_A.ndim < 2 or partition_b.ndim != 1:
            raise ValueError("Input array must have at least two dimensions and partition_b must have one dimension")

        if partition_A.shape[1] != 2 or partition_b.shape[0] != 2:
            raise ValueError("Input array must have two columns and partition_b must have two elements")
        
        x_i = partition_A[:, 1]
        w_i = partition_A[:, 0]
        X_b = partition_b[1]

        X_A = np.matmul(w_i, x_i)
        omega_A = np.sum(w_i)
        
        if np.isnan(omega_A) or omega_A == 0:
            raise ValueError(f"Invalid value of omega_A: {omega_A}")
        
        mu = X_A/omega_A
        
        XX_A = np.sum(w_i*(x_i - mu)**2)

        XX_Ab = XX_A + (((X_A - (omega_A*X_b))**2)/(omega_A*(omega_A + 1)))
        omega = omega_A + partition_b[0]

        return XX_Ab/omega

    except ValueError as ve:
        print("Error:", ve)
        
        return None
