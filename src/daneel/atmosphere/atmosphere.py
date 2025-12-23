import matplotlib.pyplot as plt
from ipywidgets import *
import numpy as np
import sys

import taurex.log
from taurex.cache import OpacityCache,CIACache
from taurex.temperature import Isothermal
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry
from taurex.chemistry import ConstantGas
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import RayleighContribution
from taurex.binning import FluxBinner,SimpleBinner
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.optimizer.nestle import NestleOptimizer



class Atmosphere:
	def __init__(self, input_params):
		self.input_params = input_params
		OpacityCache().clear_cache()
		OpacityCache().set_opacity_path(input_params['file_path']['opacity_path'])
		CIACache().set_cia_path(input_params['file_path']['cia_path'])
		self.planet = Planet(planet_radius=input_params['planet']['planet_radius'],planet_mass=input_params['planet']['planet_mass'])
		self.star = BlackbodyStar(temperature=input_params['star']['temperature'],radius=input_params['star']['radius'])
		self.chemistry = TaurexChemistry(fill_gases=input_params['chemistry']['fill_gases']['gas'],ratio=input_params['chemistry']['fill_gases']['ratio'])
		for key in input_params['chemistry']['ConstantGas']:
			self.chemistry.addGas(ConstantGas(key,mix_ratio=input_params['chemistry']['ConstantGas'][key]))
		self.isothermal =Isothermal(T = input_params['temperature']['isothermal'])
		self.min_pre = input_params['pressure']['min']
		self.max_pre = input_params['pressure']['max']
		self.lay = input_params['pressure']['layers']
		self.cia_par = input_params['cia_pairs']
		self.mod_data_path = input_params['save_path']['model']['data']
		self.mod_para_path = input_params['save_path']['model']['para']
		self.mod_fig_path = input_params['save_path']['model']['fig']
		
		self.ret_data_path = input_params['file_path']['data_path']
		self.fit_fig_path = input_params['save_path']['retrieve']['fig']
		self.ret_para_path = input_params['save_path']['retrieve']['para']
	
	def model(self):
		tm = TransmissionModel(planet=self.planet,
						temperature_profile=self.isothermal,
						chemistry=self.chemistry,
						star=self.star,
						atm_min_pressure=self.min_pre,
						atm_max_pressure=self.max_pre,
						nlayers=self.lay)
		tm.add_contribution(AbsorptionContribution())
		tm.add_contribution(CIAContribution(cia_pairs=self.cia_par))
		tm.add_contribution(RayleighContribution())
		tm.build()
		res = tm.model()
		wngrid = np.sort(10000/np.logspace(-0.4,1.1,1000))
		bn = SimpleBinner(wngrid=wngrid)

		bin_wn, bin_rprs,_,_  = bn.bin_model(tm.model(wngrid=wngrid))
		errorbars = np.full_like (bin_rprs,0.00001)

		Data = np.zeros((len(bin_wn),3))
		Data [:,0] = 10000/bin_wn
		Data [:,1] = bin_rprs
		Data [:,2] = errorbars
		output_df = np.savetxt (self.mod_data_path , Data, header='Wavelength(micron) (rp/rs)^2 Error_bars')
		
		with open(self.mod_para_path, "w") as f:
			f.write("Planet: XO-2N b\n")
			f.write("Model: TauREx Transmission Spectrum\n\n")
			f.write("Planet parameters:\n")
			f.write(f"  Radius: {self.input_params['planet']['planet_radius']:.2f} Rj\n")
			f.write(f"  Mass:   {self.input_params['planet']['planet_mass']:.2f} Mj\n\n")
			f.write("Star parameters:\n")
			f.write(f"  Temperature: {self.input_params['star']['temperature']:.2f} K\n")
			f.write(f"  Radius:      {self.input_params['star']['radius']:.2f} Rsun\n\n")
			
			f.write("Atmosphere:\n")
			for key in self.input_params['chemistry']['ConstantGas']:
				f.write(f"  {key}: {self.input_params['chemistry']['ConstantGas'][key]:.2e}\n")
			
			f.write("Error model:\n")
			f.write("  Constant 10 ppm\n")
			
		
		binned_fig = plt.figure()
		
		plt.plot(10000/bin_wn,bin_rprs)
		plt.xscale('log')
		plt.savefig(self.mod_fig_path)
		


	def retrieve(self):
		tm = TransmissionModel(planet=self.planet,
						temperature_profile=self.isothermal,
						chemistry=self.chemistry,
						star=self.star,
						atm_min_pressure=self.min_pre,
						atm_max_pressure=self.max_pre,
						nlayers=self.lay)
		tm.add_contribution(AbsorptionContribution())
		tm.add_contribution(CIAContribution(cia_pairs=self.cia_par))
		tm.add_contribution(RayleighContribution())
		tm.build()
		res = tm.model()
		obs = ObservedSpectrum(self.ret_data_path)
		
		opt = NestleOptimizer(num_live_points=50)
		
		#Setting up the model and observed spectrum for the optimizer
		opt.set_model(tm)
		opt.set_observed(obs)
		
		#Set up which parameters to fit and their boundaries
		
		for key in self.input_params['fit_params']:
			opt.enable_fit(key)
			opt.set_boundary(key,self.input_params['fit_params'][key])
		
		fit_output = opt.fit()
		taurex.log.disableLogging()
		obin = obs.create_binner()
		
		for fit_output,optimized_map,optimized_value,values in opt.get_solution():
			opt.update_model(optimized_map)
			ax = plt.figure()
			plt.errorbar(obs.wavelengthGrid,obs.spectrum,obs.errorBar,label='Obs',color='k',markersize=2,alpha=0.5,elinewidth=0.25)
			plt.plot(obs.wavelengthGrid,obin.bin_model(tm.model(obs.wavenumberGrid))[1],label='TM')
			plt.scatter(obs.wavelengthGrid, obs.spectrum, label='Obs', c='C3', s=2)
			plt.legend()
			
			
		
		
		ax.savefig(self.fit_fig_path)
		
		with open(self.ret_para_path, 'w') as f:
			# Save Input (Fixed) Parameters
			f.write("================ INPUT / FIXED PARAMETERS ================\n")
			f.write(f"Planet Mass:     {tm.planet.mass}\n")
			f.write(f"Star Temperature:{tm.star.temperature}\n")
			f.write(f"Star Radius:     {tm.star.radius}\n")
			f.write(f"Min Pressure:    {tm.pressure.profile[-1]}\n")
			f.write(f"Max Pressure:    {tm.pressure.profile[0]}\n")
			f.write(f"Num Layers:      {tm.nLayers}\n")
			f.write("\n")
			# Save Retrieved Parameters with Error Bars
			f.write("================ RETRIEVED PARAMETERS ================\n")
			f.write(f"{'Parameter':<20} {'Median':<15} {'+1sigma':<15} {'-1sigma':<15} {'Best-Fit (Map)':<15}\n")
			f.write("-" * 85 + "\n")
			# Iterate through the solution
			
			for sol_idx, map, median, extra in opt.get_solution():
				# Update model with best fit parameters for the plot later
				opt.update_model(map)
				samples = opt.get_samples(sol_idx)
				param_names = opt.fit_names
				
				for i, name in enumerate(param_names):
					s = samples[:, i]
					# Calculate 16th, 50th (Median), and 84th percentiles
					q16, q50, q84 = np.percentile(s, [16, 50, 84])
					sigma_plus = q84 - q50
					sigma_minus = q50 - q16
					best_fit_val = median[i]
					# Write formatted line to file
					f.write(f"{name:<20} {q50:<15.5e} {sigma_plus:<15.5e} {sigma_minus:<15.5e} {best_fit_val:<15.5e}\n")
		
		
		
		
		
