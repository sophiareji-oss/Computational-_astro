import batman
import numpy as np
import matplotlib.pyplot as plt

params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 2.615838                       #orbital period
params.rp = 0.103                       #planet radius (in units of stellar radii)
params.a = 7.993                        #semi-major axis (in units of stellar radii)
params.inc = 88.01                      #orbital inclination (in degrees)
params.ecc = 0.028                       #eccentricity
params.w = 261                        #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.4984, 0.0785]      #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(-0.1, 0.1, 1000)  #times at which to calculate light curve
m = batman.TransitModel(params, t)    #initializes model

flux = m.light_curve(params)

plt.plot(t, flux)
plt.ylabel("Relative Flux")
plt.xlabel("time from central transit (days)")
plt.title("Light curve of XO-2N b transit")
plt.savefig('XO-2N b_assignment1_taskF.png')
