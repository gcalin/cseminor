import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import linregress

def compute_averaged_values():
    """Computes the average values of multiple simulations.
    """
    pass


def read_file_lines(filename, cols, skip=0, stop=-1, column_major=False, separator=' '):
    """Reads real values from the columns from a file.

    Args:
        filename ([string]): The name of the file. 
        cols ([int]): Which columns to select.
        skip (int, optional): How many lines to skip in the beginning (if any). Defaults to 0.
        stop (int, optional): At which line to stop (if any). -1 means to read till the end.
        column_major (bool, optional): Wehteher to return the the values per column (True) or per line (False). Defaults to False.
        separator (str, optional): The string which separates values on a column. Defaults to ' '.

    Returns:
        [[numpy.float64]]: A list of the read values.
    """

    # Set current directory
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    # Open file
    f = open(__location__ + '/' + filename, "r")

    # Read lines and skip initial lines if necessary
    lines = f.readlines()[skip:]

    # Select columns
    res = [[np.float64(line[col]) for col in cols] for line in [l.strip().split(separator) for l in lines]]

    return np.transpose(res) if column_major else res

def plot(title, xlabel, ylabel, grid, vals, labels, loglog=True):
    """Plots multiple sets of values on a common grid.

    Args:
        title (title of the plot): The title of the plot.
        xlabel (string): The label for the x axis.
        ylabel (string): The label for the y axis.
        grid ([float]): The common grid on which the values should be plotted.
        vals ([float): A list of sets containing the values to be plotted.
        labels ([string]): A list of names for each set of values.
        loglog (bool, optional): Whether or not to plot the values on adoubly logarithmic scale. Defaults to True.
    """

    assert len(labels) == len(vals)
    

    # Plot for each pair of values and labels
    for value, label in zip(vals, labels):

        # Set up plot
        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Compute the slope for a particular set of values
        slope = linregress(grid, value).slope

        # Generate a random (RGB) color. For common style, we should probably remove this in the future
        random_color = (random.random(), random.random(), random.random())

        # Plot the values and the slope
        if loglog:
            plt.loglog(grid, value, c= random_color, label=label, marker='o')
            plt.loglog(grid, slope*grid, '--', c=random_color, label = "slope of " + label)
            
        else:
            plt.plot(grid, value, c=(random.random(), random.random(), random.random()), label=label)
            plt.plot(grid, slope*grid, '--', label = "slope of " + label)

        plt.legend()        
        plt.show()

def plot_diffusivity():
    lines = read_file_lines('selfdiffusivity.dat', [0, 1, 2], skip=3, column_major=True)
    
    plot('Diffusivity!', 'Time', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['Hydrogen', 'Oxygen'])


def plot_viscosity():
    lines = read_file_lines('viscosity.dat', [0, 8, 9], skip=3, column_major=True)
    plot('Plot of viscosity!', 'Time', 'Viscosity', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'])

plot_diffusivity()

plot_viscosity()

