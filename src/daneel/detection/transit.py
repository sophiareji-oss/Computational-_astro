import batman
import numpy as np
import matplotlib.pyplot as plt

class Transit:
	def __init__(self, input_params):
		self.input_params = input_params
		self.params = batman.TransitParams()       #object to store transit parameters
		self.params.name = input_params['name']    # the name of the planet
		self.params.t0 = input_params['t0']                #time of inferior conjunction
		self.params.per = input_params['per']                #orbital period
		self.params.rp = input_params['rp']              #planet radius (in units of stellar radii)
		self.params.a = input_params['a']                   #semi-major axis (in units of stellar radii)
		self.params.inc = input_params['inc']                      #orbital inclination (in degrees)
		self.params.ecc = input_params['ecc']                    #eccentricity
		self.params.w = input_params['w']                        #longitude of periastron (in degrees)
		self.params.limb_dark = input_params['limb_dark']        #limb darkening model
		self.params.u = input_params['u']
	
	def flux(self):
		t = np.linspace(-0.2, 0.2, 1000)  #times at which to calculate light curve
		m = batman.TransitModel(self.params, t)    #initializes model

		flux = m.light_curve(self.params)

		plt.plot(t, flux)
		plt.ylabel("Relative Flux")
		plt.xlabel("time from central transit (days)")
		plt.title("Light curve of {0} transit".format(self.params.name))
		plt.savefig('{}_assignment1_taskF.png'.format(self.params.name))
