import numpy as np
import matplotlib.pyplot as plt

x = [1, 2]

density = [0.9623, 0.9948]
density_error = [0.0005, 0.0004]
density_ref = [0.964, 0.999]

# dimension 1: 1m, 2m, 4m
# dimension 2: H2O, Na, Cl, CO2
self_diff = np.array([[9.7, 4.92, 6.06, 8.4],
                      [8.4, 4.36, 5.19, 7.3]])
self_diff_error = np.array([[0.2, 0.07, 0.09, 0.2],
                            [0.1, 0.09, 0.09, 0.1]])
self_diff_ref_CO2 = np.array([7.22, 5.27])

# dimension 1: 1m, 2m, 4m
# dimension 2: shear, bulk
visc = np.array([[26.6, 1.2],
                 [32, 0.72]])#*10^-5
visc_error = np.array([[0.9, 0.1],
                       [1, 0.08]])#*10^-5

# dimension 1: 1m, 2m, 4m
# dimension 2: H2O, Na, Cl, CO2
MS_diff = np.array([[4.97, 6.12, 8.5, 3.12, 4.33, 5.3],
                    [4.44, 5.29, 7.4, 2.75, 3.84, 4.6]])
MS_diff_error = np.array([[0.07, 0.09, 0.2, 0.08, 0.04, 0.1],
                          [0.09, 0.09, 0.1, 0.08, 0.07, 0.1]])
MS_diff_ref_CO2 = np.array([7.24, 7.61])
MS_diff_ref_CO2_error = np.array([4, 4])
MS_diff_ref2_CO2 = np.array([7.20, 7.25])
MS_diff_ref2_CO2_error = np.array([1.6, 5])

cond = [14.0, 23.9]
cond_error = [0.2, 0.3]
cond_ref = [16.3, 29.7]
cond_ref_error = [4.1, 11.0]

plt.figure("Density")
plt.clf()
plt.plot(x, density_ref, '--')
plt.errorbar(x, density, density_error, capsize = 3)
plt.ylabel("density ($g/cm^3$)")
plt.xlabel("molality ($m$)")
#plt.ylim((0.9,1.1))
plt.legend(("MD reference", "measurements"))

plt.figure("Self diffusivity")
plt.clf()
plt.plot(x, self_diff_ref_CO2, '--')
for i in range(len(self_diff[0])):
    plt.errorbar(x, self_diff[:,i], self_diff_error[:,i], capsize = 3)
plt.ylabel("self-diffusivity ($10^{-9} m^2/s$)")
plt.xlabel("molality ($m$)")
plt.legend(("$CO_2$ MD reference", "$H_2O$", "$Na^+$", "$Cl^-$", "$CO_2$"))

# only plot shear viscosity
plt.figure("Viscosity")
plt.clf()
plt.errorbar(x, visc[:,0], visc_error[:,0], capsize = 3)
plt.errorbar(x, visc[:,1], visc_error[:,1], capsize = 3)
plt.ylabel("viscosity ($10^{-5} Pa s$)")
plt.xlabel("molality ($m$)")
plt.legend(("shear", "bulk"))

plt.figure("MS diffusivity")
plt.clf()
for i in range(len(MS_diff[0])):
    plt.errorbar(x, MS_diff[:,i], MS_diff_error[:,i], capsize = 3)
plt.ylabel("MS-diffusivity ($10^{-9} m^2/s$)")
plt.xlabel("molality ($m$)")
plt.legend(("$D^{MS}_{H_2O-Na^+}$", "$D^{MS}_{H_2O-Cl^-}$", "$D^{MS}_{H_2O-CO_2}$", "$D^{MS}_{Na^+-Cl^-}$", "$D^{MS}_{Na^+-CO_2}$", "$D^{MS}_{Cl^--CO_2}$"), mode="expand", ncol=6, bbox_to_anchor=(0, 1.07, 1, 0.05))

plt.figure("MS diffusivity reference")
plt.clf()
plt.plot(x, MS_diff_ref_CO2, ls='--')
plt.plot(x, MS_diff_ref2_CO2, ls='--')
plt.errorbar(x, MS_diff[:,2], MS_diff_error[:,2], capsize = 3)
plt.errorbar(x, MS_diff[:,4], MS_diff_error[:,4], capsize = 3)
plt.errorbar(x, MS_diff[:,5], MS_diff_error[:,5], capsize = 3)
plt.ylabel("MS-diffusivity ($10^{-9} m^2/s$)")
plt.xlabel("molality ($m$)")
plt.legend(("$D^{MS}_{CO_2-brine}$ MD reference Einstein", "$D^{MS}_{CO_2-brine}$ MD reference Green-Kubo", "$D^{MS}_{H_2O-CO_2}$", "$D^{MS}_{Na^+-CO_2}$", "$D^{MS}_{Cl^--CO_2}$"), mode="expand", ncol=6, bbox_to_anchor=(0, 1.07, 1, 0.05))

plt.figure("Ionic conductivity")
plt.clf()
plt.plot(x, cond_ref, ls='--')
plt.errorbar(x, cond, cond_error, capsize = 3)
plt.ylabel("Ionic conductivity ($S/m$)")
plt.xlabel("molality ($m$)")
plt.legend(("MD reference Einstein", "MD reference Green-Kubo", "measurements"))

