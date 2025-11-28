import batman
import numpy as np
import matplotlib.pyplot as plt

class Transit:
	def __init__(self, input_params):
		self.input_params = input_params
		self.params = batman.TransitParams()                    #object to store transit parameters
	
	def flux(self):
		for key in self.input_params['transit']:
			
			self.params = batman.TransitParams()                    #object to store transit parameters
			self.params.t0 = self.input_params['transit'][key]['t0']                     #time of inferior conjunction
			self.params.per = self.input_params['transit'][key]['per']                   #orbital period
			self.params.rp = self.input_params['transit'][key]['rp']                     #planet radius (in units of stellar radii)
			self.params.a = self.input_params['transit'][key]['a']                       #semi-major axis (in units of stellar radii)
			self.params.inc = self.input_params['transit'][key]['inc']                   #orbital inclination (in degrees)
			self.params.ecc = self.input_params['transit'][key]['ecc']                   #eccentricity
			self.params.w = self.input_params['transit'][key]['w']                       #longitude of periastron (in degrees)
			self.params.limb_dark = self.input_params['transit'][key]['limb_dark']       #limb darkening model
			self.params.u = self.input_params['transit'][key]['u']
			

			t = np.linspace(-0.2, 0.2, 1000)  #times at which to calculate light curve
			m = batman.TransitModel(self.params, t)    #initializes model

			flux = m.light_curve(self.params)

			plt.plot(t, flux,label="{} planet transit".format(key))
			plt.ylabel("Relative Flux")
			plt.xlabel("time from central transit (days)")
			plt.title("Light curve of the transit")
			
		plt.legend()
		plt.savefig('transit_lightcurve.png')
