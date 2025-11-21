import batman
import numpy as np
import matplotlib.pyplot as plt

class Transit:
	def __init__(self, input_params):
		self.input_params = input_params
		self.params = batman.TransitParams()                    #object to store transit parameters
		self.params.t0 = input_params['transit']['t0']                     #time of inferior conjunction
		self.params.per = input_params['transit']['per']                   #orbital period
		self.params.rp = input_params['transit']['rp']                     #planet radius (in units of stellar radii)
		self.params.a = input_params['transit']['a']                       #semi-major axis (in units of stellar radii)
		self.params.inc = input_params['transit']['inc']                   #orbital inclination (in degrees)
		self.params.ecc = input_params['transit']['ecc']                   #eccentricity
		self.params.w = input_params['transit']['w']                       #longitude of periastron (in degrees)
		self.params.limb_dark = input_params['transit']['limb_dark']       #limb darkening model
		self.params.u = input_params['transit']['u']
	
	def flux(self):
		t = np.linspace(-0.2, 0.2, 1000)  #times at which to calculate light curve
		m = batman.TransitModel(self.params, t)    #initializes model

		flux = m.light_curve(self.params)

		plt.plot(t, flux)
		plt.ylabel("Relative Flux")
		plt.xlabel("time from central transit (days)")
		plt.title("Light curve of the transit")
		plt.savefig('assignment1_taskF.png')
