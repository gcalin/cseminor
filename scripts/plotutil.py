import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import linregress
import re

def compute_averaged_values(filenames, cols, skip=0, stop=-1, column_major=False, separator='[\t ]+'):
    """Computes the average values of multiple simulations.

    Args:
        filenames ([string]): The names of the files to read.
        cols ([int]): The columns to consider in each file.
        skip (int, optional): How many lines to skip in the beginning (if any). Defaults to 0.
        stop (int, optional): At which line to stop (if any). -1 means to read till the end.
        column_major (bool, optional): Wehteher to return the the values per column (True) or per line (False). Defaults to False.
        separator (str, optional): The string which separates values on a line. Defaults to '[\t ]+'.

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

def read_file_lines(filename, cols, skip=0, stop=-1, column_major=False, separator='[\t ]+'):
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
    lines = f.readlines()[skip:stop]

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
paths = ['./../data/H2O/run1/', './../data/H2O/run2/', './../data/H2O/run3/', './../data/H2O/run4/', './../data/H2O/run5/']
slurm = ['slurm-2585572.out', 'slurm-2586424.out', 'slurm-2586498.out', 'slurm-2586501.out', 'slurm-2586508.out']

def plot_density():
    density = np.zeros(len(paths))
    # Read the lines and plot the results
    data_lines = [[685,786], [685,786], [685,786], [685,786], [51,552]] #lines may be different per slurm file.
    for i in range(len(paths)):
        file = paths[i] + slurm[i]
        lines = read_file_lines(file, [0, 11], skip=data_lines[i][0], stop=data_lines[i][1], column_major=True)
        #plot('Plot of Density '+str(i+1), 'Timestep', 'Density (g/cm$^3$)', lines[0], lines[1:], ['Density'], loglog=False, linear=None, show_slope=False)
        density[i]=np.mean(lines[1])
    return density

def plot_total_energy():

    # Particular filenames for total energy
    filenames = [path + 'TotalEnergy.dat' for path in paths]
    # Read the lines and plot the results
    for i in range(len(filenames)):
        file = filenames[i]
        lines = read_file_lines(file, [0, 1], skip=2, column_major=True)
        plot('Plot of Total Energy '+str(i+1), 'Timestep', 'Total Energy (Kcal/mol)', lines[0], lines[1:], ['Total Energy'], loglog=False, linear=None, show_slope=False)

def plot_diffusivity():

    # Particular filenames for diffusivity
    filenames = [path + 'selfdiffusivity.dat' for path in paths]
    linearParts = [[[14,36],[11,36]], [[15,36],[13,36]], [[18,36],[14,36]], [[14,27],[12,27]], [[11,41],[11,41]]] #[None, None, None]
    self_diff = np.zeros((len(filenames), 2))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        if (i < 3):
            lines = read_file_lines(file, [0, 1, 2], skip=3, column_major=True)
        else: # run 4 and 5 have their data in a differetn order.
            lines = read_file_lines(file, [0, 2, 1], skip=3, column_major=True)
        self_diff[i,:]=plot('Plot of diffusivity '+str(i+1), 'Time', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['Hydrogen', 'Oxygen'], linear=linearPart) 
    return self_diff

def plot_viscosity():

    # Particular filenames for viscosity
    filenames = [path + 'viscosity.dat' for path in paths]
    linearParts = [[[29,44],[29,39]], [[30,49],[31,40]], [[30,42],[27,49]], [[34,48],[30,39]], [[30,42],[27,49]]] #[None, None, None]
    visc = np.zeros((len(filenames), 2))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 8, 9], skip=3, column_major=True)
        visc[i,:]=plot('Plot of viscosity '+str(i+1), 'Time', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'], linear=linearPart)
    return visc

def plot_rdf():
    filenames = [path + 'rdf.dat' for path in paths]
    #lines = read_file_lines('./../data/H2O/run1/rdf.dat', [0, 2, 4, 6], skip=1, column_major=True)
    lines = compute_averaged_values(filenames, [0, 2, 4, 6], skip=1, column_major=True)
    plot('Plot of radial distribution function (rdf)', 'Radius', r'Density', lines[0], lines[1:], ['Hydrogen-Hydrogen', 'Hydrogen-Oxygen', 'Oxygen-Oxygen'], show_slope=False)
    
density = plot_density() #uncomment print statement in function for density plots.
#plot_total_energy() #uncomment for energy plots.

N=800 #number of molecules
self_diff = plot_diffusivity()
for i in range(len(self_diff)):
    if ((self_diff[i] !=  None).all()):
        self_diff[i][0] = self_diff[i][0]/(2*N) #Hydrogen
        self_diff[i][1] = self_diff[i][1]/N #Oxygen

T=303.15 #temperature
visc = plot_viscosity()
for i in range(len(visc)):
    if ((visc[i] != None).all()):
        visc[i][0]=visc[i][0]/T #shear viscosity
        visc[i][1]=visc[i][1]/T #bulk viscosity

kB = 1.38064852e-23 #m^2 kg s^-2 K^-1
T = 298.15 #temperature in K
xi = 2.837298 #for periodic (cubic) lattices
m_water = (2*1.00794+15.9994)  #mass of a water molecule
N_avogadro = 6.02214076e23 
m = N*(m_water)/N_avogadro
V = m/density * 10**-6 #m^3
L = V**(1/3) #m

# Self diffusivity correction
for i in range(len(self_diff[0])):
    self_diff[:,i] += kB*T*xi/(6*np.pi*(visc[:,0]*1.01325e-10)*L) * 1e5

#calculate average and standard deviation
avg_dens = np.average(density, 0)
std_dens = np.std(density, 0)
print("Average density: %10.3e +/-%10.3e (%3.2f%%) g/cm^3." %(avg_dens, std_dens, std_dens/avg_dens*100))

avg_diff = np.average(self_diff, 0)
std_diff = np.std(self_diff, 0)

print("Self-diffusion constant of Hydrogen:%10.3e +/-%10.3e (%3.2f%%) angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[0], std_diff[0], std_diff[0]/avg_diff[0]*100))
print("Self-diffusion constant of Oxygen:%10.3e +/-%10.3e (%3.2f%%) angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[1], std_diff[1], std_diff[1]/avg_diff[1]*100))

avg_visc = np.average(visc, 0)
std_visc = np.std(visc, 0)
# documantation: https://www.geeksforgeeks.org/numpy-std-in-python/

print("Shear viscosity of water:%10.3e +/-%10.3e (%3.2f%%) atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[0],std_visc[0],std_visc[0]/avg_visc[0]*100))
print("Bulk viscosity of water:%10.3e +/-%10.3e (%3.2f%%) atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[1],std_visc[1],std_visc[1]/avg_visc[1]*100))

plot_rdf()

print("\nFound vs expected results")
print("Density =%10.3e vs%10.3e" %(avg_dens,0.99705))
print("Viscosity =%10.3e vs%10.3e" %(avg_visc[0]*(1.01325e-10),890.08e-6))
# data for comparison from NIST database



