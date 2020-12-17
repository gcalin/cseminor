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
    all_cols = read_file_lines(filenames, cols, skip, stop, column_major, separator)

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

    #this needs to be outside of the for loop, otherwise not all graphs will be plotted.
    plt.figure(title)
    plt.clf()

    # Plot for each pair of values and labels
    for i in range(len(labels)):
        value = vals[i]
        label = labels[i]

        # Compute the slope for a particular set of values
        if linear == None:
            slope = linregress(grid, value).slope
        else:
            #linear regression on the linear part of the graph
            start = linear[i][0]
            end = linear[i][1]
            slope = linregress(grid[start:end], value[start:end]).slope
            slopes.append(slope)

        # Generate a random (RGB) color. For common style, we should probably remove this in the future
        random_color = (random.random(), random.random(), random.random())
        print(slopes)

        # Set up plot
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Plot the values and the slope
        if loglog:
            if (value[0]>0):
                plt.loglog(grid, value, c=random_color, label=label, marker='o')
                if show_slope:
                    plt.loglog(grid, slope*grid, '--', c=random_color, label = "slope of " + label)
                
            else:
                plt.loglog(grid, np.abs(value), c=random_color, label=label+'(negative)', marker='o')
                if show_slope:
                    plt.loglog(grid, np.abs(slope*grid), '--', c=random_color, label = "slope of " + label)
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
paths = './../data/NaCl/'

def plot_total_energy():

    # Particular filenames for diffusivity
    filenames = paths + 'TotalEnergy.dat'
    #linearParts = [[20,40],[33,36]]#, [[15,36],[13,36]], [[18,36],[14,36]], [[7,41],[9,41]]] #[None, None, None]
    #self_diff = np.zeros((1, 2))
    #file = filenames
    #linearPart = linearParts[i]
    # Read the lines and plot the results
    lines = read_file_lines(filenames, [0, 1], skip=3, column_major=True)
    self_diff=plot('Plot of Total Energy ', 'Time', 'Total Energy', lines[0], lines[1:], ['Total Energy'], loglog=False, show_slope=False)
    #return self_diff

def plot_diffusivity():

    # Particular filenames for diffusivity
    filenames = paths + 'selfdiffusivity.dat'
    linearParts = [[20,40],[33,36]]#, [[15,36],[13,36]], [[18,36],[14,36]], [[7,41],[9,41]]] #[None, None, None]
    self_diff = np.zeros((1, 2))
    file = filenames
    #linearPart = linearParts[i]
    # Read the lines and plot the results
    lines = read_file_lines(file, [0, 1, 2], skip=3, column_major=True)
    self_diff=plot('Plot of diffusivity ', 'Time', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['water', 'NaCl'], linear=linearParts)
    return self_diff

def plot_viscosity():

    # Particular filenames for viscosity
    filenames = paths + 'viscosity.dat'
    linearParts = [[19,59],[42,49]]#, [[30,49],[31,40]], [[30,42],[27,49]], [[30,42],[27,49]]] #[None, None, None]
    visc = np.zeros((1, 2))
    file = filenames
    #linearPart = linearParts[i]
    # Read the lines and plot the results
    lines = read_file_lines(file, [0, 8, 9], skip=3, column_major=True)
    visc=plot('Plot of viscosity ', 'Time', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'], linear=linearParts)
    return visc

def plot_MS_diffusivity():

    # Particular filenames for MS diffusivity
    filenames = paths + 'onsagercoefficient.dat'
    linearParts = [[21,31],[13,29],[11,29]]#, [[30,49],[31,40]], [[30,42],[27,49]], [[30,42],[27,49]]] #[None, None, None]
    MS_diff = np.zeros((1, 3))
    file = filenames
    #linearPart = linearParts[i]
    # Read the lines and plot the results
    lines = read_file_lines(file, [0, 1, 2, 3], skip=3, column_major=True)
    MS_diff = plot('Plot of MS diffusivity ', 'Time', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['water-water', 'water-NaCl', 'NaCl-NaCl'], linear=linearParts)
    return MS_diff

plot_total_energy()  

self_diff = plot_diffusivity()
#print(self_diff)
if (self_diff != None):
    self_diff[0] = self_diff[0]/3000 #water
    self_diff[1] = self_diff[1]/54 #NaCl

T=298.15 #temperature
visc = plot_viscosity()
#print(visc)
if ((visc != None)):
    visc[0]=visc[0]/T #shear viscosity
    visc[1]=visc[1]/T #bulk viscosity

kb=1.38064852e-23 # Boltzmann's constant
e=2.837298 #Constant
L=7.20198e-24*1.01325 # Length of the box (also including units adaptation)
correct_self_diff=[0,0]
correct_self_diff[0] = self_diff[0] + (kb*T*e)/(6*np.pi*visc[0]*L)
correct_self_diff[1] = self_diff[1] + (kb*T*e)/(6*np.pi*visc[0]*L)

N=3054 #number of molecules
mf_water=3000/N # mole fraction
mf_NaCl=54/N # mole fraction
MS_diff = plot_MS_diffusivity()
#print(MS_diff)
if ((MS_diff != None)):
    MS_diff = mf_NaCl/mf_water*MS_diff[0] + mf_water/mf_NaCl*MS_diff[2]-2*MS_diff[1] #MS_diffusivity
"""   
# calculate average and standard deviation
avg_diff = np.average(self_diff, 0)
avg_visc = np.average(visc, 0)

std_diff = np.std(self_diff, 0)
std_visc = np.std(visc, 0)
# documantation: https://www.geeksforgeeks.org/numpy-std-in-python/
"""
print("Self-diffusion constant of water:",self_diff[0],"angstrom^2/femtosecond = 10^-5 m^2/s.")
print("Self-diffusion constant of NaCl:",self_diff[1],"angstrom^2/femtosecond = 10^-5 m^2/s." )
print("Corrected self-diffusion constant of water:",correct_self_diff[0],"angstrom^2/femtosecond = 10^-5 m^2/s.")
print("Corrected self-diffusion constant of NaCl:",correct_self_diff[1],"angstrom^2/femtosecond = 10^-5 m^2/s." )

print("Shear viscosity of the system:",visc[0],"atm*femtoseconds = 1.01325·10^−10 Pas.")
print("Bulk viscosity of the system:",visc[1],"atm*femtoseconds = 1.01325·10^−10 Pas.")

print("MS_Diffusivity of the system:",MS_diff,"angstrom^2/femtosecond = 10^-5 m^2/s.")

"""
print("Self-diffusion constant of water:%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[0], std_diff[0]))
print("Self-diffusion constant of NaCl:%10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[1], std_diff[1]))

print("Shear viscosity of the system:%10.3e +/-%10.3e atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[0],std_visc[0]))
print("Bulk viscosity of the system:%10.3e +/-%10.3e atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[1],std_visc[1]))
"""

