import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.stats import linregress
import re

'''
Adaptations required for new data files:
    Adapt paths to use the correct datafiles.
    Make sure the correct lines and columns are read.
    Adapt number of molecules and temperature.
    Adapt text of the print statements.
    Run the code using None for the linear parts of the graphs.
    Determine the linear parts of the graphs and add them to the code.
    Run the code again to find the resutls.
'''

def read_file_lines(filename, cols, skip=0, stop=-1, column_major=False, separator='[\t ]+'):
    """Reads real values from the columns from a file.

    Args:
        filename ([string]): The name of the file. 
        cols ([int]): Which columns to select.
        skip (int, optional): How many lines to skip in the beginning (if any). Defaults to 0.
        stop (int, optional): At which line to stop (if any). -1 means to read till the end.
        column_major (bool, optional): Wehteher to return the the values per column (True) or per line (False). Defaults to False.
        separator (str, optional): The string which separates values on a line. Defaults to '[\t ]+'.

    Returns:
        [[numpy.float64]]: A list of the read values.
    """

    # Set current directory
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
    # Open file
    f = open(__location__ + "/" + filename, "r")

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
        linear ([int], optional): A list of sets (one set for every graph) containing the starting and ending indices of the linear part of the graph.
        show_slope (bool, optional): Whether or not to display the slope of the plot. Defaults to True.
        
    Returns:
        slopes ([float], optinal) OR None: if linear is not None, the slopes of the linear parts are returned.
        If linear part is None, then None is returned.
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
paths = ['./../data/CO2/Restart1/']
slurm = ['slurm-1789.out']

def plot_density():
    density = np.zeros(len(paths))
    # Read the lines and plot the results
    data_lines = [[81,132]] #lines may be different per slurm file.
    for i in range(len(paths)):
        file = paths[i] + slurm[i]
        lines = read_file_lines(file, [0, 11], skip=data_lines[i][0], stop=data_lines[i][1], column_major=True)
        plot('Plot of Density '+str(i+1), 'Timestep', 'Density (g/cm$^3$)', lines[0], lines[1:], ['Density'], loglog=False, linear=None, show_slope=False)
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
    linearParts = [[[5,-1], [10,-2], [6,19]]] #[None, None, None]
    self_diff = np.zeros((len(filenames), 3))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 1, 2, 3], skip=3, column_major=True)
        self_diff[i,:]=plot('Plot of diffusivity '+str(i+1), 'Time (fs)', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['H2O', 'NaCl', 'CO2'], linear=linearPart)   
    return self_diff

def plot_viscosity():

    # Particular filenames for viscosity
    filenames = [path + 'viscosity.dat' for path in paths]
    linearParts = [[[13,36], [30,37]]] #[[[29,44],[29,39]], [[30,49],[31,40]], [[30,42],[27,49]], [[30,42],[27,49]]] #[None, None, None]
    visc = np.zeros((len(filenames), 2))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 8, 9], skip=3, column_major=True)
        visc[i,:]=plot('Plot of viscosity '+str(i+1), 'Time (fs)', r'$MSD_{Viscosity}$', lines[0], lines[1:], ['MSD_all', 'MSD_bulkvisc'], linear=linearPart)
    return visc

def plot_onsager_coef():

    # Particular filenames for diffusivity
    filenames = [path + 'onsagercoefficient.dat' for path in paths]
    linearParts = [[[9,18],[0,-1],[0,-1],[1,19],[0,-1],[3,18]]] #[[[14,36],[11,36]], [[15,36],[13,36]], [[18,36],[14,36]], [[7,41],[9,41]]] #[None, None, None]
    onsager_coef = np.zeros((len(filenames), 6))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 1, 2, 3, 4, 5, 6], skip=2, column_major=True)
        lines = lines + abs(np.amin(lines))+1 # +1 to make sure all values are >0, and not equal to 0.
        onsager_coef[i,:]=plot('Plot of onsager coefficients '+str(i+1), 'Time (fs)', r'$MSD_{Onsager}$', lines[0], lines[1:], ['H2O-H2O', 'H2O-NaCl', 'H2O-CO2', 'NaCl-NaCl', 'NaCl-CO2', 'CO2-CO2'], linear=linearPart, loglog=True)   
    return onsager_coef

density = plot_density()
plot_total_energy()

#number of atoms for 1m
N_water = 3000 # = 1000 molecules
N_NaCl  = 36   # = 18 molecules
N_CO2   = 18   # = 6 molecules
N = N_water + N_NaCl + N_CO2 #total number of atoms
m_water = (2*1.00794+15.9994)/3  #avg mass of an atom in water
m_NaCl  = (22.9898+35.4530)/2    #avg mass of an atom in NaCl
m_CO2   = (12.0107+2*15.9994)/3  #avg mass on an atom in CO1
N_avogadro = 6.02214076e23 
m = (N_water*m_water + N_NaCl*m_NaCl + N_CO2*m_CO2)/N_avogadro
T = 298.15 #temperature
kB = 1.38064852e-23 #m^2 kg s^-2 K^-1
V = m/density * 10**-6 #m^3
L = V**(1/3) #m
xi = 2.837298 #for periodic (cubic) lattices
e = 1.60217662e-19 #Coulomb 
q_Na = 1
q_Cl = -1
#TODO: check units of variables above.

self_diff = plot_diffusivity()
for i in range(len(self_diff)):
    if ((self_diff[i] !=  None).all()):
        self_diff[i][0] = self_diff[i][0]/N_water #H2O
        self_diff[i][1] = self_diff[i][1]/N_NaCl #NaCl
        self_diff[i][2] = self_diff[i][2]/N_CO2 #CO2

visc = plot_viscosity()
for i in range(len(visc)):
    if ((visc[i] != None).all()):
        for j in range(len(visc[i])):
            visc[i][j]=visc[i][j]/T
# visc[i][0] is shear viscosity of run i, visc[i][1] is bulk voscosity of run i.

# Self diffusivity correction
for i in range(len(self_diff[0])):
    self_diff[:,i] += kB*T*xi/(6*np.pi*(visc[:,0]*1.01325e-10)*L) * 1e5

'''
#Calculation of MS diffusivity from onsager coefficients for ternary mixture.
onsager_coef = plot_onsager_coef()
for i in range(len(onsager_coef)):
    if ((onsager_coef[i] != None).all()):
        for j in range(len(onsager_coef[i])):
            onsager_coef[i][j]=onsager_coef[i][j]/N

## MS diffusivity for termnary mixtures
# molar fractions
x1 = N_water/N 
x2 = N_NaCl/N
x3 = N_CO2/N

MS_diff = np.zeros((len(onsager_coef), 3))

# TODO: check formulas below and possibly rewrite them.
for i in range(len(onsager_coef)): #loop over all runs
    if ((onsager_coef[i] != None).all()):
        Lambda11 = onsager_coef[i][0] # MSD_Water_Water
        Lambda12 = onsager_coef[i][1] # MSD_Water_NaCl
        Lambda13 = onsager_coef[i][2] # MSD_Water_CO2
        Lambda22 = onsager_coef[i][3] # MSD_NaCl_NaCl
        Lambda23 = onsager_coef[i][4] # MSD_NaCl_CO2
        Lambda33 = onsager_coef[i][5] # MSD_CO2_CO2
        # for every run calculate the 3 MS diffusion coefficients.
        Delta11 = (1-x1)*(Lambda11/x1-Lambda13/x3) - \
                      x1*(Lambda12/x1-Lambda23/x3 + \
                          Lambda13/x1-Lambda33/x3)
        Delta12 = (1-x1)*(Lambda12/x2-Lambda13/x3) - \
                      x1*(Lambda22/x2-Lambda23/x3 + \
                          Lambda23/x2-Lambda33/x3)
        Delta21 = (1-x2)*(Lambda12/x1-Lambda23/x3) - \
                      x2*(Lambda11/x1-Lambda13/x3 + \
                          Lambda13/x1-Lambda33/x3)
        Delta22 = (1-x2)*(Lambda22/x2-Lambda23/x3) - \
                      x2*(Lambda12/x2-Lambda13/x3 + \
                          Lambda23/x2-Lambda33/x3)
        Delta = np.array([[Delta11, Delta12], [Delta21, Delta22]])
        B = np.linalg.inv(Delta)    # B = Delta^-1
        B11 = B[0,0]
        B12 = B[0,1]
        B21 = B[1,0]
        B22 = B[1,1]
        MS_diff[i][0] = 1/(B11-(x1+x3)/x1*B12) # D12
        MS_diff[i][1] = 1/(B11+(x2/x1)*B12)    # D13
        MS_diff[i][2] = 1/(B22+(x1/x2)*B21)    # D23
'''

# molar fractions
x1 = N_water/N 
x2 = N_NaCl/N
x3 = N_CO2/N

MS_diff = np.zeros((len(self_diff), 3))
Sum = x1/self_diff[:,0]+x2/self_diff[:,1]+x3/self_diff[:,2]
# Calculation of MS diffusivity form self diffusivities.
for i in range(len(self_diff)):
    MS_diff[i][0] = self_diff[i][0]*self_diff[i][1]*Sum[i]  # D12 (water,NaCl)
    MS_diff[i][1] = self_diff[i][0]*self_diff[i][2]*Sum[i]  # D13 (water,CO2)
    MS_diff[i][2] = self_diff[i][1]*self_diff[i][2]*Sum[i]  # D23 (NaCl, CO2)

# Ionic conductivity calculation.
ion_cond = (e*e/(kB*T*V))*(1/2*N_NaCl*(q_Na)**2*self_diff[:,1] + 1/2*N_NaCl*(q_Cl)**2*self_diff[:,1])*10**-5 # Siemens/m

## TODO: find out what units to use. Is V the volume? How do we find that, take the average?
## TODO: implement this for quarternary mixture.

'''for calculating standard deviation when multiple runs are used.'''
# calculate average and standard deviation over the different runs
avg_dens = np.average(density, 0)
std_dens = np.std(density, 0)
print("Average density: %10.3e +/-%10.3e g/cm^3." %(avg_dens, std_dens))

avg_diff = np.average(self_diff, 0)
std_diff = np.std(self_diff, 0)
print("Self-diffusion constant of H2O: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[0], std_diff[0]))
print("Self-diffusion constant of NaCl: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[1], std_diff[1]))
print("Self-diffusion constant of CO2: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[2], std_diff[2]))

avg_visc = np.average(visc, 0)
std_visc = np.std(visc, 0)
print("Shear viscosity of the system: %10.3e +/-%10.3e atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[0],std_visc[0]))
print("Bulk viscosity of the system: %10.3e +/-%10.3e atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[1],std_visc[1]))

avg_MSdiff = np.average(MS_diff, 0)
std_MSdiff = np.std(MS_diff, 0)
print("MS diffusivity of Water and NaCl: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[0],std_MSdiff[0]))
print("MS diffusivity of Water and CO2: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[1],std_MSdiff[1]))
print("MS diffusivity of NaCl and CO2: %10.3e +/-%10.3e angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[2],std_MSdiff[2]))

avg_cond = np.average(ion_cond, 0)
std_cond = np.std(ion_cond, 0)
print("Electric conductivity: %10.3e +/-%10.3e S/m." %(avg_cond, std_cond))

