from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.interpolate

def find_sample_index(peak_matrix: np.array, data_original: pd.DataFrame):
    """
    Finds the sample index of the peaks in the original data.
    
    Parameters
    ----------
    peak_matrix : np.array
        A matrix of peaks with the following columns: peak number, peak index, peak value.
    
    Returns
    -------
    np.array
        The sample index of the peaks.
    """
    
    x = data_original.iloc[:, 0]
    y = data_original.iloc[:, 1]
    sample_index = np.zeros((peak_matrix.shape[0], 1))
    values = np.zeros((peak_matrix.shape[0], 1))
    

    for i in range(peak_matrix.shape[0]):
        abs_diff = np.abs(x - peak_matrix[i, 1])
        #abs_diff = np.abs(y - peak_matrix[i, 2])
        # Find the index of the closest element
        closest_indices = np.where(abs_diff == abs_diff.min())[0]
        if len(closest_indices) > 1 : 
            middle_index = len(closest_indices)//2
            closest_indices = closest_indices[middle_index]
            #closest_indices = closest_indices[0]
        values[i][0] = y[closest_indices]
        sample_index[i][0] = x[closest_indices]

        
    return sample_index, values


def moving_averages_filter(data : pd.DataFrame, window_size : int) -> pd.DataFrame:
    """
    Apply moving averages filter to the data

    Parameters
    ----------
    data : pd.DataFrame
        The data to be filtered
    window_size : int
        The size of the window

    Returns
    -------
    pd.DataFrame
        The filtered data
    """
    
    y = data.iloc[:, 1]
    x = data.iloc[:, 0]

    # Apply uniform filter for noise reduction
    smoothed = scipy.ndimage.uniform_filter1d(y, size=window_size)

    # Create a new DataFrame with the filtered data
    filtered_data = pd.DataFrame({'z_pos': x, 'contrast': smoothed})
    return filtered_data

def test_moving_averages_filter():
    """
    Test the moving_averages_filter function by creating plots to see filtered data.
    """

    data = pd.read_csv('small.csv', header=None)
    filtered_data = moving_averages_filter(data, 4)

    #plot data and smoothed data
    plt.figure(figsize=(10, 6))
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], label='Original Signal', color='blue' )
    plt.plot(filtered_data.iloc[:, 0], filtered_data.iloc[:, 1], label='Smoothed Signal', color='red')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Contrast Value')
    plt.title('Original vs. Smoothed Signal')
    plt.show()


def data_interpolation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate the data

    Parameters
    ----------
    data : pd.DataFrame
        The data to be interpolated
    
    Returns
    -------
    pd.DataFrame
        The interpolated data
    """

    y = data.iloc[:, 1]
    x = data.iloc[:, 0]

    #Interpolate the data with a quadratic function
    f = scipy.interpolate.interp1d(x, y,kind = 'quadratic') 
    
    xnew = np.arange(x.min(), x.max(), 0.1) #TODO bu 0.1den emin degilem degistirebilirim
    ynew = f(xnew)
    df = pd.DataFrame({'x': xnew, 'y': ynew})
    return df

def test_data_interpolation():
    """
    Test the data_interpolation function by creating plots to see interpolated data.
    """

    data = pd.read_csv('small.csv')
    df = data_interpolation(data)

    # plot data and interpolated data
    plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'o', df.iloc[:, 0], df.iloc[:, 1], '-')
    plt.show()

def find_max_index(data : pd.DataFrame) -> float:
    """
    Find the index of the maximum value in a DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with a single column.
    
    Returns
    -------
    float
        The index of the maximum value.
    """
    
    y = data.iloc[:,1]
    x = data.iloc[:,0]

    max_y = y[0]
    max_index = 0

    for i in range (len(y)):
        if y[i] > max_y:
            max_y = y[i]
            max_index = i
    
    return x[max_index]
    

def find_window_size(data : pd.DataFrame) -> int:
    """
    Find the window size for the moving averages filter by using the SAD (Sum of Absolute Differences).

    Parameters
    ----------
    data : pd.DataFrame
        The data for which the window size will be found
    
    Returns
    -------
    int
        The window size
    """


    # Extract contrast values
    y = data.iloc[:, 1]

    # Determine a list of window sizes
    window_sizes = np.arange(2,20,1)
    sad = np.array([])

    # For each window size, calculate the smoothed data and SAD of the smoothed data and original data
    for window_size in window_sizes:
        smoothed = scipy.ndimage.uniform_filter1d(y, size=window_size)
        sad = np.append(sad,np.sum(np.abs(smoothed - y)))

    # Find the index of the first value in SAD that is greater than 80% of the last value in SAD
    # This index is the index where smoothing provides no further benefit 
    # This index will be used to find the corresponding window size
    threshold = 0.8 * sad[-1]  
    index = np.where(sad > threshold)[0][0]

    # Get the corresponding windowSize
    windowSize = window_sizes[index]

    return windowSize 



