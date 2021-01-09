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
    f = open(__location__ + "/" + filename, "r")

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

        # Set up plot
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
paths = ['./../data/CO2/run7/']

def plot_diffusivity():

    # Particular filenames for diffusivity
    filenames = [path + 'selfdiffusivity.dat' for path in paths]
    linearParts = [None, None, None] #[[[14,36],[11,36]], [[15,36],[13,36]], [[18,36],[14,36]], [[7,41],[9,41]]] #[None, None, None]
    self_diff = np.zeros((len(filenames), 3))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 1, 2, 3], skip=3, column_major=True)
        self_diff[i,:]=plot('Plot of diffusivity '+str(i+1), 'Time', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['H2O', 'NaCl', 'CO2'], linear=linearPart)   
    return self_diff

def plot_viscosity():

    # Particular filenames for viscosity
    filenames = [path + 'viscosity.dat' for path in paths]
    linearParts = [None, None] #[[[29,44],[29,39]], [[30,49],[31,40]], [[30,42],[27,49]], [[30,42],[27,49]]] #[None, None]
    visc = np.zeros((len(filenames), 2))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 8, 9], skip=3, column_major=True)
        visc[i,:]=plot('Plot of viscosity '+str(i+1), 'Time', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'], linear=linearPart)
    return visc

def plot_onsager_coef():

    # Particular filenames for diffusivity
    filenames = [path + 'onsagercoefficient.dat' for path in paths]
    linearParts = [None, None, None, None, None, None] #[[[14,36],[11,36]], [[15,36],[13,36]], [[18,36],[14,36]], [[7,41],[9,41]]] #[None, None, None, None, None, None]
    onsager_coef = np.zeros((len(filenames), 6))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 1, 2, 3, 4, 5, 6], skip=2, column_major=True)
        lines = lines + abs(np.amin(lines))+1 # +1 to make sure all values are >0, and not equal to 0.
        onsager_coef[i,:]=plot('Plot of onsager coefficients '+str(i+1), 'Time', r'$MSD_{Onsager}$', lines[0], lines[1:], ['H2O-H2O', 'H2O-NaCl', 'H2O-CO2', 'NaCl-NaCl', 'NaCl-CO2', 'CO2-CO2'], linear=linearPart, loglog=True)   
    return onsager_coef

#number of molecules
N_water = 1000
N_NaCl = 18
N_CO2 = 6
N = N_water + N_NaCl + N_CO2 #total number of molecules

self_diff = plot_diffusivity()
for i in range(len(self_diff)):
    if ((self_diff[i] !=  None).all()):
        self_diff[i][0] = self_diff[i][0]/N_water #H2O
        self_diff[i][1] = self_diff[i][1]/N_NaCl #NaCl
        self_diff[i][2] = self_diff[i][2]/N_CO2 #CO2

T=298.15 #temperature
visc = plot_viscosity()
for i in range(len(visc)):
    if ((visc[i] != None).all()):
        for j in range(len(visc[i])):
            visc[i][j]=visc[i][j]/T
# visc[i][0] is shear viscosity of run i, visc[i][1] is bulk voscosity of run i.

onsager_coef = plot_onsager_coef()
for i in range(len(onsager_coef)):
    if ((onsager_coef[i] != None).all()):
        for j in range(len(onsager_coef[i])):
            onsager_coef[i][j]=onsager_coef[i][j]/N
'''
for i in range(len(onsager_coef)):
    if ((MS_diff != None)):
        MS_diff = mf_NaCl/mf_water*onsager_coef[0] + mf_water/mf_NaCl*onsager_coef[2]-2*onsager_coef[1] #MS_diffusivity
'''

## MS diffusivity for termnary mixtures

# molar fractions
x1 = N_water/N 
x2 = N_NaCl/N
x3 = N_CO2/N

MS_diff = np.zeros((len(onsager_coef), 3))

# TODO: check formulas below and possibly rewrite them.
for i in range(len(onsager_coef)): #loop over all runs
    if ((onsager_coef[i] != None).all()):
        # for every run calculate the 3 MS diffusion coefficients.
        Delta11 = (1-x1)*(onsager_coef[i][0]/x1-onsager_coef[i][2]/x3) - \
            x1*(onsager_coef[i][1]/x1-onsager_coef[i][4]/x3 + \
                onsager_coef[i][2]/x1-onsager_coef[i][5]/x3)
        Delta12 = (1-x1)*(onsager_coef[i][1]/x2-onsager_coef[i][2]/x3) - \
            x1*(onsager_coef[i][3]/x2-onsager_coef[i][4]/x3 + \
                onsager_coef[i][4]/x2-onsager_coef[i][5]/x3)
        Delta21 = (1-x2)*(onsager_coef[i][1]/x1-onsager_coef[i][4]/x3) - \
            x2*(onsager_coef[i][0]/x1-onsager_coef[i][2]/x3 + \
                onsager_coef[i][2]/x1-onsager_coef[i][5]/x3)
        Delta22 = (1-x2)*(onsager_coef[i][3]/x2-onsager_coef[i][4]/x3) - \
            x2*(onsager_coef[i][1]/x2-onsager_coef[i][2]/x3 + \
                onsager_coef[i][4]/x2-onsager_coef[i][5]/x3)
        Delta = np.array([[Delta11, Delta12], [Delta21, Delta22]])
        B = np.linalg.inv(Delta)    # B = Delta^-1
        B11 = B[0,0]
        B12 = B[0,1]
        B21 = B[1,0]
        B22 = B[1,1]
        MS_diff[i][0] = 1/(B11-(x1+x3)/x1*B12) # D12
        MS_diff[i][1] = 1/(B11+(x2/x1)*B12)    # D13
        MS_diff[i][2] = 1/(B22+(x1/x2)*B21)    # D23

'''for calculating standard deviation when multiple runs are used.'''
# calculate average and standard deviation over the different runs
avg_diff = np.average(self_diff, 0)
std_population_diff = np.std(self_diff, 0)
# documantation: https://www.geeksforgeeks.org/numpy-std-in-python/
std_sample_diff = np.std(self_diff, 0, ddof=1)
# numpy uses population standard deviation by default.
# If you want to use it to calculate sample standard deviation, 
# use an additional parameter, called ddof and set it to 1.
# https://honingds.com/blog/python-standard-deviation/
print("Self-diffusion constant of H2O: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[0], std_sample_diff[0]))
print("Self-diffusion constant of NaCl: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[1], std_sample_diff[1]))
print("Self-diffusion constant of CO2: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[2], std_sample_diff[2]))

avg_visc = np.average(visc, 0)
std_population_visc = np.std(visc, 0)
std_sample_visc = np.std(visc, 0, ddof=1)
print("Shear viscosity of the system: %10.3e +/-%10.3e atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[0],std_sample_visc[0]))
print("Bulk viscosity of the system: %10.3e +/-%10.3e atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[1],std_sample_visc[1]))

avg_MSdiff = np.average(MS_diff, 0)
std_population_MSdiff = np.std(MS_diff, 0)
std_sample_MSdiff = np.std(MS_diff, 0, ddof=1)
print("MS diffusivity of Water and NaCl: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[0],std_sample_MSdiff[0]))
print("MS diffusivity of Water and CO2: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[1],std_sample_MSdiff[1]))
print("MS diffusivity of NaCl and CO2: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[2],std_sample_MSdiff[2]))
# TODO: check units MS diffusivity.

#print("MS_Diffusivity of the system:",MS_diff,"angstrom^2/femtosecond = 10^-5 m^2/s.")


