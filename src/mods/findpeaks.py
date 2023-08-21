import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .utils import data_interpolation
from scipy.ndimage import uniform_filter1d
import math

def findpeaks(data : pd.DataFrame, slope_threshold : np.float64 ,amp_threshold: np.float64 ,smooth_width:int ,peak_group:int):
    """
    Find all peaks in given data that are above the given thresholds 
    via derivative analysis and polynomial fitting

    Parameters
    ----------
    data : pd.DataFrame
        The data provided
    slope_threshold : np.float64
        The threshold for the derivative
    amp_threshold : np.float64
        The threshold for the amplitude
    smooth_width : int
        The size of the window for smoothing the derivative
    peak_group : int
        The size of the window for polynomial fitting

    Returns
    -------
    P : np.ndarray
        A matrix containing the peaks found with columns: peak number, 
        peak index(z-position), peak value(contrast)
    fitted : np.ndarray
        The fitted curve
    peak_x : np.float64
        The x value of the peak
    xxs : list
        A list of x values(interpolated) of the sub-group of points near peak
    yys : list
        A list of y values(interpolated) of the sub-group of points near peak
    xxs_original : np.ndarray
        A list of x values(original) of the sub-group of points near peak
    yys_original : np.ndarray
        A list of y values(original) of the sub-group of points near peak

    """
    
    data_original = data
    
    data = data_interpolation(data)

    x_original = data_original.iloc[:, 0]
    y_original = data_original.iloc[:, 1]
    

    y = data.iloc[:, 1] # y values of interpolated data
    x = data.iloc[:, 0] # x values of interpolated data

    

    if smooth_width > 1:
        # Smooth derivative by moving averages for a given window size
        d = apply_smooth(np.gradient(y,0.5), smooth_width) 
        
    else:
        # For smooth_width = 1, do not apply smoothing
        d = np.gradient(y,0.5)

    n=round(peak_group/2+1) # will be used for creating sub-group of points near peak

    # Create an empty matrix for recording peaks found
    # Each row will contain the following information: peak number, peak index, peak value
    P = np.empty((0,3))

    vectorlength=len(y)
    peak = 0 # peak number
    peak_x=0 # peak index
    peak_y=0 # peak value

    j_start = 2*smooth_width//2-1
    j_end = len(y)-smooth_width-1

    xxs = []
    yys = []
    fits = []
    xxs_original = np.array([])
    yys_original = np.array([])
    index_original = np.array([],dtype=int)

    # Detect downward zero-crossing of the derivative
    for j in range(j_start-1, j_end):
        # if derivative changes sign from + to - and its value is greater than slope_threshold
        if np.sign(d[j]) > np.sign(d[j+1]) and  d[j]-d[j+1] > slope_threshold: 

            # Create sub-group of points near peak 
            
            xx=np.zeros(peak_group)
            yy=np.zeros(peak_group)
            for k in range(0, peak_group):
                groupindex = j+k-n+1
                
                if groupindex < 1:
                    groupindex = 1
                elif groupindex > vectorlength:
                    groupindex = vectorlength
                xx[k] = x[groupindex]
                yy[k] = y[groupindex]
            
            #find closest value in x_original to xx
            index = np.zeros(peak_group,dtype=int)
            closest_x = np.zeros(peak_group)
            for i in range(0,len(xx)):
                index[i],closest_x[i] = find_index(x_original,xx[i])
            closest_y = y_original[index].values

                
            if peak_group > 2:

                fitted , peak_x = poly_fit(closest_x,closest_y)

        
                peak_y = fitted[np.argmax(fitted)] #max value of fitted curve
                fits.append(fitted)
            else:
                peak_y = max(yy)
                pindex=find_index(yy,peak_y)
                peak_x=xx[pindex[0]][0]


            #constructs matrix of peaks
            if math.isnan(peak_x) or math.isnan(peak_y) or peak_y < amp_threshold:
                pass
            else:
                P = np.vstack((P, np.array([round(peak),peak_x, peak_y])))
                
                xxs.append(xx)
                yys.append(yy)
                peak = peak+1

                xxs_original = np.append(xxs_original,closest_x)
                yys_original = np.append(yys_original,closest_y)
        
    
    return P, fitted, peak_x , xxs , yys , xxs_original , yys_original 

def find_index(x, val):
    dif=np.absolute(x-val)
    
    min_val = min(dif)
    min_val_index = np.where(dif == min_val)[0]
    
    closestval = x[min_val_index]
    return min_val_index,closestval


                    

def apply_smooth(y:np.ndarray , window_size:int) -> np.ndarray:
    """
    Apply moving averages filter to given data

    Parameters
    ----------
    y : np.ndarray
        The data to be smoothed
    window_size : int
        The size of the window
    """

    #Smooth data by moving averages
    smooth_y = uniform_filter1d(y, size=window_size) 
    return smooth_y


def poly_fit(x,y):
    """
    Fit a polynomial of degree 3 to the given data
    """
    # p is a vector of coefficients p that minimises the squared error
    p = np.polyfit(x, y, 3) 

    xp = np.linspace(min(x), max(x), 100)

    fitted = np.polyval(p, xp) # values of polynomial p evaluated at xp
    max_at = xp[np.argmax(fitted)] # find the index of max value of fitted

    return fitted, max_at

