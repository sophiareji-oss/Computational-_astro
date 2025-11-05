import batman
import numpy as np
import matplotlib.pyplot as plt

params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 2.615838                 #orbital period
params.rp = 0.973                       #planet radius (in units of stellar radii)
params.a = 0.0369                        #semi-major axis (in units of stellar radii)
params.inc = 88.7                      #orbital inclination (in degrees)
params.ecc = 0.0456                       #eccentricity
params.w = 261.0                        #longitude of periastron (in degrees)
params.limb_dark = "quadratic"        #limb darkening model
params.u = [0.4984,0.0785]      #limb darkening coefficients [u1, u2, u3, u4]

t = np.linspace(-0.075, 0.075, 1000)  #times at which to calculate light curve
m = batman.TransitModel(params, t)    #initializes model

flux = m.light_curve(params)

plt.plot(t, flux)
plt.savefig('XO-2N_b_assignment1_taskF.png')

