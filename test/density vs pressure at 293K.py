import matplotlib.pyplot as plt

pressure = [20,80,140,200]
real = [0.013848386844,0.0617350895061,0.115652918465,0.1641108264]
simulation = [0.0133709,0.05300035,0.0931775,0.133335]

line1, = plt.plot(pressure,real,'b-o')
line1.set_label('NIST database')
line2, = plt.plot(pressure,simulation,'r-o')
line2.set_label('simulation data')
plt.legend()
plt.show()
