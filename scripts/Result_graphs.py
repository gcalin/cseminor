import numpy as np
import matplotlib.pyplot as plt

x = [1, 2]

density = [0.9623, 0.9948]
density_error = [0.0005, 0.0004]
density_ref = [0.964, 0.999]

# dimension 1: 1m, 2m, 4m
# dimension 2: H2O, Na, Cl, CO2
self_diff = np.array([[9.7, 4.92, 6.06, 8.4],
                      [8.36, 3.50, 4.3, 11.5]])
self_diff_error = np.array([[0.2, 0.07, 0.09, 0.2],
                            [0.08, 0.08, 0.2, 0.1]])

# dimension 1: 1m, 2m, 4m
# dimension 2: shear, bulk
visc = np.array([[2.66e-04, 1.2e-05],
                 [3.2e-04, 6.9e-06]])
visc_error = np.array([[9e-06, 1e-06],
                       [1e-05 , 7e-07]])

# dimension 1: 1m, 2m, 4m
# dimension 2: H2O, Na, Cl, CO2
MS_diff = np.array([[4.97, 6.12, 8.5, 3.12, 4.33, 5.3],
                    [3.59, 4.4, 11.8, 1.86, 4.9, 6.1]])
MS_diff_error = np.array([[0.07, 0.09, 0.2, 0.08, 0.04, 0.1],
                          [0.08, 0.2, 0.1, 0.07, 0.1, 0.2]])

cond = [14.0, 19.6]
cond_error = [0.2, 0.4]
cond_ref = [16.3, 29.7]
cond_ref2 = [14.6, 30.5]

plt.figure("Density")
plt.clf()
plt.plot(x, density_ref, '--')
plt.errorbar(x, density, density_error, capsize = 3)
plt.ylabel("density ($g/cm^3$)")
plt.xlabel("molality ($m$)")
plt.legend(("MD reference", "measurements"))

plt.figure("Self diffusivity")
plt.clf()
for i in range(len(self_diff[0])):
    plt.errorbar(x, self_diff[:,i], self_diff_error[:,i], capsize = 3)
plt.ylabel("self-diffusivity ($m^2/s$)")
plt.xlabel("molality ($m$)")
plt.legend(("$H_2O$", "$Na^+$", "$Cl^-$", "$CO_2$"))

plt.figure("Viscosity")
plt.clf()
for i in range(len(visc[0])):
    plt.errorbar(x, visc[:,i], visc_error[:,i], capsize = 3)
plt.ylabel("viscosity ($Pa s$)")
plt.xlabel("molality ($m$)")
plt.legend(("shear", "bulk"))

plt.figure("MS diffusivity")
plt.clf()
for i in range(len(MS_diff[0])):
    plt.errorbar(x, MS_diff[:,i], MS_diff_error[:,i], capsize = 3)
plt.ylabel("MS-diffusivity ($m^2/s$)")
plt.xlabel("molality ($m$)")
plt.legend(("$H_2O-Na^+$", "$H_2O-Cl^-$", "$H_2O-CO_2$", "$Na^+-Cl^-$", "$Na^+-CO_2$", "$Cl^--CO_2$"))

plt.figure("Ionic conductivity")
plt.clf()
plt.plot(x, cond_ref, '--')
plt.plot(x, cond_ref2, '--')
plt.errorbar(x, cond, cond_error, capsize = 3)
plt.ylabel("Ionic conductivity ($S/m$)")
plt.xlabel("molality ($m$)")
plt.legend(("MD reference Einstein", "MD reference Green-Kubo", "measurements"))

