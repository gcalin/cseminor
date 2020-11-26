import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import linregress
import re

def compute_averaged_values():
    """Computes the average values of multiple simulations.
    """
    pass


def read_file_lines(filename, cols, skip=0, stop=-1, column_major=False, separator='[\t ]'):
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
    res = [[np.float64(line[col]) for col in cols] for line in [re.split(separator, l.strip()) for l in lines]]
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
    
    # Set up plot
    plt.figure(title)
    plt.clf()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Plot for each pair of values and labels
    for value, label in zip(vals, labels):

        # Compute the slope for a particular set of values
        slope = linregress(grid, value).slope

        # Generate a random (RGB) color. For common style, we should probably remove this in the future
        random_color = (random.random(), random.random(), random.random())

        # Plot the values and the slope
        if loglog:
            plt.loglog(grid, value, c= random_color, label=label, marker='o')
            plt.loglog(grid, slope*grid, '--', c=random_color, label = "slope of " + label)
            
        else:
            plt.plot(grid, value, c= random_color, label=label, marker='o')
            plt.plot(grid, slope*grid, '--', c=random_color, label = "slope of " + label)

        plt.legend()        
        plt.show()

def plot_diffusivity():
    lines = read_file_lines('./../data/H2O/selfdiffusivity.dat', [0, 1, 2], skip=3, column_major=True)
    plot('Plot of diffusivity', 'Time', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['Hydrogen', 'Oxygen'])

def plot_viscosity():
    lines = read_file_lines('./../data/H2O/viscosity.dat', [0, 8, 9], skip=3, column_major=True)
    plot('Plot of viscosity', 'Time', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'])

def plot_rdf():
    lines = read_file_lines('./../data/H2O/rdf.dat', [0, 2, 4, 6], skip=1, column_major=True)
    plot('Plot of radial distribution function (rdf)', 'Radius', r'Density', lines[0], lines[1:], ['Hydrogen-Hydrogen', 'Hydrogen-Oxygen', 'Oxygen-Oxygen'])
    

plot_diffusivity()

plot_viscosity()

plot_rdf()

