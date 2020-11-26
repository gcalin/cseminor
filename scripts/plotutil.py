import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import linregress
import re

def compute_averaged_values(filenames, cols, skip=0, stop=-1, column_major=False, separator='[\t ]'):
    """Computes the average values of multiple simulations.

    Args:
        filenames ([string]): The names of the files to read.
        cols ([int]): The columns to consider in each file.
        skip (int, optional): How many lines to skip in the beginning (if any). Defaults to 0.
        stop (int, optional): At which line to stop (if any). -1 means to read till the end.
        column_major (bool, optional): Wehteher to return the the values per column (True) or per line (False). Defaults to False.
        separator (str, optional): The string which separates values on a line. Defaults to '[\t ]'.

    Returns:
        [type]: A set of columns with the value of each entry equal to the average of the the entries in each file.
    """

    # Read columns
    all_cols = [read_file_lines(file, cols, skip, stop, column_major, separator) for file in filenames]

    # Initialise results array
    res = [np.zeros(len(i)) for i in all_cols[0]]

    # For each set of columns
    for file_data in all_cols:
        # For each column of the file
        for num, col in enumerate(file_data):
            # Add the value divided by the amount of files considered (to get the average)
            res[num] += col/len(all_cols)

    return res

def compute_errors_per_column(filenames, cols, skip=0, stop=-1, column_major=True, separator='[\t ]'):
    """Computes the error of the entries in a column across multiple files.

    Args:
        filenames ([string]): The names of the files to read.
        cols ([int]): The columns to consider in each file.
        skip (int, optional): How many lines to skip in the beginning (if any). Defaults to 0.
        stop (int, optional): At which line to stop (if any). -1 means to read till the end.
        column_major (bool, optional): Wehteher to return the the values per column (True) or per line (False). Defaults to False.
        separator (str, optional): The string which separates values on a line. Defaults to '[\t ]'.

    Returns:
        [type]: A set of columns with the entries on each column calculated as 100*stddev/mean across that entry in multiple files.
    """

    # Read the files
    all_cols = [read_file_lines(file, cols, skip, stop, column_major, separator) for file in filenames]

    # Initialise results array
    res = [[[] for j in all_cols[0][0]] for i in all_cols[0]]

    # For each set of columns
    for file_data in all_cols:
        # For each column in the file
        for i, col in enumerate(file_data):
            # For each value in the column
            for j, val in enumerate(col):
                # Append the value to a tuple in the results array
                res[i][j].append(val)

    # Return the error at each time step across all columns
    return [[np.std(valtuple)*100/np.average(valtuple) for valtuple in col] for col in res]

def read_file_lines(filename, cols, skip=0, stop=-1, column_major=False, separator='[\t ]'):
    """Reads real values from the columns from a file.

    Args:
        filename ([string]): The name of the file. 
        cols ([int]): Which columns to select.
        skip (int, optional): How many lines to skip in the beginning (if any). Defaults to 0.
        stop (int, optional): At which line to stop (if any). -1 means to read till the end.
        column_major (bool, optional): Wehteher to return the the values per column (True) or per line (False). Defaults to False.
        separator (str, optional): The string which separates values on a line. Defaults to '[\t ]'.

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

def plot(title, xlabel, ylabel, grid, vals, labels, loglog=True, linear=None, show_slope=True):
    """Plots multiple sets of values on a common grid.

    Args:
        title (title of the plot): The title of the plot.
        xlabel (string): The label for the x axis.
        ylabel (string): The label for the y axis.
        grid ([float]): The common grid on which the values should be plotted.
        vals ([float]): A list of sets containing the values to be plotted.
        labels ([string]): A list of names for each set of values.
        loglog (bool, optional): Whether or not to plot the values on a doubly logarithmic scale. Defaults to True.
        linear ([int], optional): A list of sets containing the starting and ending indices of the linear part of the graph.
        show_slope (bool, optional): Whether or not to display the slope of the plot. Defaults to True.
        
    Returns:
        slopes ([float], optinal): contains the slopes of the linear parts if linear is not None.
    """

    assert len(labels) == len(vals)
    if linear != None:
        assert len(labels) == len(linear)
    
    slopes = []

    # Plot for each pair of values and labels
    for i in range(len(labels)):
        value = vals[i]
        label = labels[i]

        # Compute the slope for a particular set of values
        if linear == None:
            slope = linregress(grid, value).slope
        else:
            start = linear[i][0]
            end = linear[i][1]
            slope = linregress(grid[start:end], value[start:end]).slope
            slopes.append(slope)

        # Generate a random (RGB) color. For common style, we should probably remove this in the future
        random_color = (random.random(), random.random(), random.random())

        # Set up plot
        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Plot the values and the slope
        if loglog:
            plt.loglog(grid, value, c=random_color, label=label, marker='o')
            if show_slope:
                plt.loglog(grid, slope*grid, '--', c=random_color, label = "slope of " + label)
            
        else:
            plt.plot(grid, value, c=random_color, label=label, marker='o')
            if show_slope:
                plt.plot(grid, slope*grid, '--', c=random_color, label = "slope of " + label)

        plt.legend()        
        plt.show()
    
    if linear != None:
        return slopes
    else:
        return None

# TODO: remove global variable in the future
paths = ['./../data/H2O/run1/', './../data/H2O/run2/', './../data/H2O/run3/']

def plot_diffusivity():

    # Particular filenames for diffusivity
    filenames = [path + 'selfdiffusivity.dat' for path in paths]
    #lines = read_file_lines('./../data/H2O/run1/selfdiffusivity.dat', [0, 1, 2], skip=3, column_major=True)

    # Read the lines and plot the results
    lines = compute_averaged_values(filenames, [0,1,2], skip=3, column_major=True)
    self_diff = plot('Plot of diffusivity', 'Time', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['Hydrogen', 'Oxygen'], linear=[[15,len(lines[0])], [11,len(lines[0])]])
    
    # Get the array of errors at each point and plot
    err = compute_errors_per_column(filenames, [1,2], skip=3, column_major=True)
    plot('Errors in diffusivity', 'Time', r'100*stddev/mean', np.log10(lines[0]), err, ['Hydrogen', 'Oxygen'], loglog=False, linear=None, show_slope=False)
    return self_diff

def plot_viscosity():

    # Particular filenames for viscosity
    filenames = [path + 'viscosity.dat' for path in paths]
    #lines = read_file_lines('./../data/H2O/run1/viscosity.dat', [0, 8, 9], skip=3, column_major=True)   

    # Read the lines and plot the results
    lines = compute_averaged_values(filenames, [0,8,9], skip=3, column_major=True)
    visc = plot('Plot of viscosity', 'Time', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'], linear=[[23,39], [24,36]])

    # Get the array of errors at each point and plot
    err = compute_errors_per_column(filenames, [8,9], skip=3, column_major=True)
    plot('Errors in viscosity', 'Time', r'100*stddev/mean', np.log10(lines[0]), err, ['MSD_all', 'MSD_bulkvisc'], loglog=False, linear=None, show_slope=False)
    return visc

def plot_rdf():
    filenames = [path + 'rdf.dat' for path in paths]
    #lines = read_file_lines('./../data/H2O/run1/rdf.dat', [0, 2, 4, 6], skip=1, column_major=True)
    lines = compute_averaged_values(filenames, [0, 2, 4, 6], skip=1, column_major=True)
    plot('Plot of radial distribution function (rdf)', 'Radius', r'Density', lines[0], lines[1:], ['Hydrogen-Hydrogen', 'Hydrogen-Oxygen', 'Oxygen-Oxygen'], show_slope=False)
    
N=800 #number of molecules
self_diff = plot_diffusivity()
self_diff[0] = self_diff[0]/(2*N) #Hydrogen
self_diff[1] = self_diff[1]/N #Oxygen

T=303.15 #temperature
visc = plot_viscosity()
visc[0]=visc[0]/T #shear viscosity
visc[1]=visc[1]/T #bulk viscosity

print("Self diffusion constant of Hydrogen:", self_diff[0], "angstrom^2/femtosecond = 10^-5 m^2/s.")
print("Self diffusion constant of Oxygen:", self_diff[1], "angstrom^2/femtosecond = 10^-5 m^2/s.")
print("Shear viscosity of water:", visc[0], "atm*femtoseconds = 1.01325·10^−10 Pas." )
print("Bulk viscosity of water:", visc[1], "atm*femtoseconds = 1.01325·10^−10 Pas.")

plot_rdf()

