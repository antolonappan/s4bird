import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
import pymaster
import camb
import os
import healpy as hp
import scipy.optimize as opt
from plancklens.helpers import mpi


class cosmology:
    
    def __init__(self,lib_dir,nside,binsize):
        if mpi.rank == 0:
            os.makedirs(lib_dir,exist_ok=True)
            
        self.lib_dir = lib_dir
        self.b = pymaster.NmtBin.from_nside_linear(nside,binsize)
        self.ell = self.b.get_effective_ells()
        self.n_ell = len(self.ell)
        self.dl = self.ell*(self.ell+1)/(2*np.pi)

    def get_spectra(self,r=0):
        fname = os.path.join(self.lib_dir,f"spectra_r{r}.pkl")
        if not os.path.isfile(fname):
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
            pars.InitPower.set_params(As=2e-9, ns=0.965, r=r)
            pars.set_for_lmax(2500, lens_potential_accuracy=0)
            pars.WantTensors = True if r is not 0 else False
            print("Setting Cosmology")
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
            print("Power Spectra computed")
            if mpi.rank == 0:
                pk.dump(powers, open(fname,'wb'),protocol=2)
                print("Power spectra cached")
            return powers
            
        else:
            print("returning cache")
            return pk.load(open(fname,'rb'))

    def get_BB(self,component,r=0):
        result = self.get_spectra(r)
        if component == 'T':
            return result['tensor'].T[2]
        elif component == 'S':
            return result['lensed_scalar'].T[2]

    def get_beam(self,beam):
        fwhm = hp.gauss_beam(np.radians(beam/60.),self.b.lmax,True)[:,2]
        return self.b.bin_cell(fwhm)
    
    def get_noise(self,level):
        return np.ones(self.n_ell) * (np.radians(level/60)**2)

    def get_bandpower(self,component, r=0, in_dl=False):
        bandpower = self.get_BB(component,r)
        dl = self.dl if not in_dl else np.ones(self.n_ell)
        print("Bandpower calculated")
        return self.b.bin_cell(bandpower[:self.b.lmax + 1])  / dl

    def get_bandpower_cstm(self,fid,in_dl=False):
        dl = self.dl if in_dl else np.ones(self.n_ell)
        return self.b.bin_cell(fid[:self.b.lmax+1]) * dl


class LH_base:
    
    def __init__(self,lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed):
        self.lib_dir = lib_dir
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
        self.nsamples = nsamples
        
        self.cosmo = cosmology(self.lib_dir,512,10)
        self.tensor = self.cosmo.get_bandpower('T',r=1)
        self.lensing = self.cosmo.get_bandpower_cstm(cl_len)
        self.beam = self.cosmo.get_beam(beam)
        self.noise = self.cosmo.get_noise(nlev_p)
        
        
        
        len_m,len_s,del_m,del_s = eff_lib.get_stat
        bias = eff_lib.bias
        bias_s = eff_lib.bias_std
        self.init = [float(init[0]),float(init[1])]
        
        self.select = np.where((self.cosmo.ell >= lmin)& (self.cosmo.ell <= lmax))[0]

        if fit_lensed:
            print('Fitting Lensed spectra')
            self.mean = len_m
            self.std = len_s
        else:
            print('Fitting Delensed spectra')
            self.mean = del_m - bias
            self.std = np.sqrt(del_s**2 + bias_s**2)
        
    def chi_sq(self,theta):
        pass
    
    def initial_opt(self):
        return opt.minimize(self.chi_sq, self.init)
        
    def cl_theory(self,r,Alens):
        th = (r * self.tensor) + (Alens * self.lensing)
        return th*self.beam**2 + self.noise
    

    def log_prior(self,theta):
        r,Alens= theta
        if  -0.5 < r < 0.5 and 0 < Alens < 2.:
            return 0.0
        return -np.inf

    def log_probability(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp  -.5*self.chi_sq(theta)

    def posterior(self):
        res = self.initial_opt()
        pos = np.array(res['x']) + 1e-4 * np.random.randn(100, 2)
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        sampler.run_mcmc(pos, self.nsamples,progress=True)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return flat_samples
    
    def plot_posterior(self):
        labels = ["r","Alens"]
        flat_samples = self.posterior()
        fig = corner.corner(flat_samples, labels=labels,truths=[0,0])
    
class LH_simple(LH_base):
    
    def __init__(self,lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed=False):
        super().__init__(lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed)
        print('Likelihood: Simple')
    
    def chi_sq(self,theta):
        r,Alens = theta
        th = self.cl_theory(r,Alens)
        return np.sum(((self.mean - th)**2/self.std**2)[self.select])
    
class LH_smith(LH_base):
    
    def __init__(self,lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed=False):
        super().__init__(lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed)
        print('Likelihood: Smith')
    
    def chi_sq(self,theta):
        alpha = -1
        r,Alens = theta
        th = self.cl_theory(r,Alens)
        l = self.cosmo.ell
        _1 = (2*l) + 1
        _a = (2*l) + alpha
        _f = 9/2
        _2 = ((_1)/(_a))**(1/3.)
        
        first = _1*_f*_2
        second = ((self.mean/th)**(1/3.) -_2)**2
        third = (1-alpha)*np.log(th)
        
        chi = (first*second) + third
        
        return  np.sum(chi[self.select])