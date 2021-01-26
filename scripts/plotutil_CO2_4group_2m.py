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
    Make sure labels for plots are correct.
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
paths = ['./../data/CO2/4group/CO2_2m_30ns/run1/',
         './../data/CO2/4group/CO2_2m_30ns/run2/',
         './../data/CO2/4group/CO2_2m_30ns/run3/',
         './../data/CO2/4group/CO2_2m_30ns/run4/',
         './../data/CO2/4group/CO2_2m_30ns/run5/']

def plot_density():
    
    # Particular filenames for total energy
    filenames = [path + 'log.lammps' for path in paths]
    density = np.zeros(len(filenames))
    # Read the lines and plot the results
    for i in range(len(filenames)):
        file = filenames[i]
        lines = read_file_lines(file, [0, 11], skip=246, stop=547, column_major=True)
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
    #linearParts = [None for file in filenames]
    linearParts = [[[11,-1],[17,26],[10,-1],[11,36]],\
                   [[11,-1],[12,-1],[10,27],[12,27]],\
                   [[11,-1],[10,-1],[10,27],[11,27]],\
                   [[10,-1],[13,32],[10,29],[11,27]],\
                   [[11,-1],[7,36],[10,28],[10,27]]]
    # first dimension represents the run, second dimension the lines in every run.
    self_diff = np.zeros((len(filenames), 4))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 1, 2, 3, 4], skip=3, column_major=True)
        self_diff[i,:]=plot('Plot of diffusivity '+str(i+1), 'Time (fs)', r'$MSD_{Diffusivity}$', lines[0], lines[1:], ['H2O', 'Cl', 'CO2', 'Na'], linear=linearPart)   
    return self_diff

def plot_viscosity():

    # Particular filenames for viscosity
    filenames = [path + 'viscosity.dat' for path in paths]
    #linearParts = [None for file in filenames]
    linearParts = [[[25,45],[39,46]],\
                   [[19,45],[37,45]],\
                   [[20,45],[38,45]],\
                   [[19,45],[37,45]],\
                   [[20,45],[38,45]]]
    # first dimension represents the run, second dimension the lines in every run.
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
    linearParts = [None for file in filenames] #[[[9,18],[0,-1],[0,-1],[1,19],[0,-1],[3,18]]] #[None for file in filenames]
    onsager_coef = np.zeros((len(filenames), 6))
    for i in range(len(filenames)):
        file = filenames[i]
        linearPart = linearParts[i]
        # Read the lines and plot the results
        lines = read_file_lines(file, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], skip=2, column_major=True)
        lines = lines + abs(np.amin(lines))+1 # +1 to make sure all values are >0, and not equal to 0.
        onsager_coef[i,:]=plot('Plot of onsager coefficients '+str(i+1), 'Time (fs)', r'$MSD_{Onsager}$', lines[0], lines[1:], ['H2O-H2O', 'H2O-Cl', 'H2O-CO2', 'H2O-Na', 'Cl-Cl', 'Cl-CO2', 'Cl-Na', 'CO2-CO2', 'CO2-Na', 'Na-Na'], linear=linearPart, loglog=True)   
    return onsager_coef

density = plot_density()
#plot_total_energy()

#number of atoms for 1m
N_water = 3000 # = 1000 molecules
N_Cl    = 36   # = 36 molecules
N_CO2   = 15   # = 5 molecules
N_Na    = 36   # = 36 molecules
N = N_water + N_Cl + N_CO2 + N_Na #total number of atoms
m_water = (2*1.00794+15.9994)/3  #avg mass of an atom in water
m_Cl  = 35.4530                #mass of Cl-
m_CO2   = (12.0107+2*15.9994)/3  #avg mass on an atom in CO1
m_Na  = 22.9898                #mass of Na+
N_avogadro = 6.02214076e23 
m = (N_water*m_water + N_Cl*m_Cl + N_CO2*m_CO2 + N_Na*m_Na)/N_avogadro
T = 393.15 #temperature in K
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
        self_diff[i][1] = self_diff[i][1]/N_Cl #Cl
        self_diff[i][2] = self_diff[i][2]/N_CO2 #CO2
        self_diff[i][3] = self_diff[i][3]/N_Na #Na
    

visc = plot_viscosity()
for i in range(len(visc)):
    if ((visc[i] != None).all()):
        for j in range(len(visc[i])):
            visc[i][j]=visc[i][j]/T
# visc[i][0] is shear viscosity of run i, visc[i][1] is bulk voscosity of run i.

# Self diffusivity correction
#for i in range(len(self_diff[0])):
#    self_diff[:,i] += kB*T*xi/(6*np.pi*(visc[:,0]*1.01325e-10)*L) * 1e5

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
x_water = N_water/N 
x_Cl = N_Cl/N
x_CO2 = N_CO2/N
x_Na = N_Na/N

MS_diff = np.zeros((len(self_diff), 6))
Sum = x_water/self_diff[:,0]+x_Cl/self_diff[:,1]+x_CO2/self_diff[:,2]+x_Na/self_diff[:,3]
# Calculation of MS diffusivity form self diffusivities.
for i in range(len(self_diff)):
    MS_diff[i][0] = self_diff[i][0]*self_diff[i][1]*Sum[i]  # (water, Cl)
    MS_diff[i][1] = self_diff[i][0]*self_diff[i][2]*Sum[i]  # (water, CO2)
    MS_diff[i][2] = self_diff[i][0]*self_diff[i][3]*Sum[i]  # (water, Na)
    MS_diff[i][3] = self_diff[i][1]*self_diff[i][2]*Sum[i]  # (Cl, CO2)
    MS_diff[i][4] = self_diff[i][1]*self_diff[i][3]*Sum[i]  # (Cl, Na)
    MS_diff[i][5] = self_diff[i][2]*self_diff[i][3]*Sum[i]  # (CO2, Na)

# Ionic conductivity calculation.
ion_cond = (e*e/(kB*T*V))*(1/2*N_Na*(q_Na)**2*self_diff[:,3] + 1/2*N_Cl*(q_Cl)**2*self_diff[:,1])*10**-5 # Siemens/m

## TODO: find out what units to use. Is V the volume? How do we find that, take the average?
## TODO: implement this for quarternary mixture.

'''for calculating standard deviation when multiple runs are used.'''
# calculate average and standard deviation over the different runs
avg_dens = np.average(density, 0)
std_dens = np.std(density, 0)
print("Average density: %10.3e +/-%10.3e (%3.2f%%) g/cm^3.\n" %(avg_dens, std_dens, std_dens/avg_dens*100))

avg_diff = np.average(self_diff, 0)
std_diff = np.std(self_diff, 0)
diff_order = ["H2O", "Cl", "CO2", "Na"]
for i in range(len(avg_diff)):
    print("Self-diffusion constant of "+diff_order[i]+": %10.3e +/-%10.3e (%3.2f%%) angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_diff[i], std_diff[i], std_diff[i]/avg_diff[i]*100))
print("")

avg_visc = np.average(visc, 0)
std_visc = np.std(visc, 0)
print("Shear viscosity of the system: %10.3e +/-%10.3e (%3.2f%%) atm*femtoseconds = 1.01325·10^−10 Pas." %(avg_visc[0],std_visc[0],std_visc[0]/avg_visc[0]*100))
print("Bulk viscosity of the system: %10.3e +/-%10.3e (%3.2f%%) atm*femtoseconds = 1.01325·10^−10 Pas.\n" %(avg_visc[1],std_visc[1],std_visc[1]/avg_visc[1]*100))

avg_MSdiff = np.average(MS_diff, 0)
std_MSdiff = np.std(MS_diff, 0)
MSdiff_order = [["H2O","Cl"], ["H2O", "CO2"], ["H2O", "Na"], ["Cl", "CO2"], ["Cl", "Na"], ["CO2", "Na"]]
for i in range(len(avg_MSdiff)):
    print("MS diffusivity of "+MSdiff_order[i][0]+" and "+MSdiff_order[i][1]+": %10.3e +/-%10.3e (%3.2f%%) angstrom^2/femtosecond = 10^-5 m^2/s." %(avg_MSdiff[i],std_MSdiff[i],std_MSdiff[i]/avg_MSdiff[i]*100))
print("")

avg_cond = np.average(ion_cond, 0)
std_cond = np.std(ion_cond, 0)
print("Electric conductivity: %10.3e +/-%10.3e (%3.2f%%) S/m." %(avg_cond, std_cond, std_cond/avg_cond*100))

