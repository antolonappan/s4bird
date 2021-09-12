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
    
    def __init__(self,lib_dir,nside,binsize,ini,cache):
        if mpi.rank == 0:
            os.makedirs(lib_dir,exist_ok=True)
            
        self.lib_dir = lib_dir
        self.b = pymaster.NmtBin.from_nside_linear(nside,binsize)
        self.ell = self.b.get_effective_ells()
        self.n_ell = len(self.ell)
        self.dl = self.ell*(self.ell+1)/(2*np.pi)
        self.ini = ini
        self.cache = cache

    def get_spectra(self,r=0):
        fname = os.path.join(self.lib_dir,f"spectra_r{r}.pkl")
        if os.path.isfile(fname) and self.cache: 
            print("returning cache")
            return pk.load(open(fname,'rb'))
        else:
            if r == 0:
                print('Computing Scalar power spectra')
                pars = camb.read_ini(self.ini)
                results = camb.get_results(pars)
                powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
            else:
                print('Computing Tensor power spectra')
                pars = camb.read_ini(self.ini)
                pars.WantTensors = True 
                pars.InitPower.set_params(r=r)
                results = camb.get_results(pars)
                powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
                
            if mpi.rank == 0 and self.cache:
                pk.dump(powers, open(fname,'wb'),protocol=2)
                print("Power spectra cached")
            return powers


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
        ini = '/project/projectdirs/litebird/simulations/S4BIRD/CAMB/CAMB.ini'
        self.lib_dir = lib_dir
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
        self.nsamples = nsamples
        
        self.cosmo = cosmology(self.lib_dir,512,10,ini,True)
        self.tensor = self.cosmo.get_bandpower('T',r=1)
        self.lensing = self.cosmo.get_bandpower_cstm(cl_len)
        self.beam = self.cosmo.get_beam(beam)
        self.noise = self.cosmo.get_noise(nlev_p)
        self.fit_lensed = fit_lensed
        
        
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
        
        self.name = None
        
    def chi_sq(self,theta):
        pass
    
    def initial_opt(self):
        return opt.minimize(self.chi_sq, self.init)
        
    def cl_theory(self,r,Alens):
        th = (r * self.tensor) + (Alens * self.lensing)
        return th*self.beam**2 + self.noise
    

    def log_prior(self,theta):
        r,Alens= theta
        if  -0.5 < r < 0.5 and 0 < Alens < 1.5:
            return 0.0
        return -np.inf

    def log_probability(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp  -.5*self.chi_sq(theta)

    def posterior(self):
        fname = os.path.join(self.lib_dir,f"posterior_{self.name}_{self.nsamples}_L{int(self.fit_lensed)}S_{self.select[0]}_{self.select[-1]}.pkl")
        if not os.path.isfile(fname):
            #res = self.initial_opt()
            pos = np.array(self.init) + 1e-4 * np.random.randn(100, 2)
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
            sampler.run_mcmc(pos, self.nsamples,progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pk.dump(flat_samples, open(fname,'wb'),protocol=2)
            return flat_samples
        else:
            return pk.load(open(fname,'rb'))
    
    def sigma_r(self):
        samples = self.posterior()
        r_samp = np.sort(samples[:,0])
        r_pos = r_samp[r_samp>0]
        return f"{r_pos[int(len(r_pos)*.683)]:.2e}"
    
    def plot_posterior(self):
        fname = os.path.join(self.lib_dir,f"posterior_{self.name}_{self.nsamples}_L{int(self.fit_lensed)}S_{self.select[0]}_{self.select[-1]}.png")
        labels = ["r","Alens"]
        flat_samples = self.posterior()
        plt.figure(figsize=(8,8))
        fig = corner.corner(flat_samples, labels=labels,truths=[0,0])
        if not os.path.isfile(fname):
            plt.savefig(fname,bbox_inches='tight')

    
class LH_simple(LH_base):
    
    def __init__(self,lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed=False):
        super().__init__(lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed)
        print('Likelihood: Simple')
        self.name = 'simple'
    
    def chi_sq(self,theta):
        r,Alens = theta
        th = self.cl_theory(r,Alens)
        return np.sum(((self.mean - th)**2/self.std**2)[self.select])
    
class LH_smith(LH_base):
    
    def __init__(self,lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed=False):
        super().__init__(lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed)
        print('Likelihood: Smith')
        self.name = 'smith'
    
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
        
        chi = 10*(first*second) + third
        
        return  np.sum(chi[self.select])


class LH_HL(LH_base):
    def __init__(self,lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed=False,basename=None):
        super().__init__(lib_dir,eff_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,init,fit_lensed)
        print('Likelihood: HL')
        self.name = 'HL'
        self.fileselector_del = {'LiteBird':['c_fid.pkl','cov_del.pkl'],
                           'litebird_cmbs4':['c_fid_s4.pkl','cov_del_s4.pkl'],
                           'litebird_S4pLB':['c_fid_s4plb.pkl','cov_del_s4plb.pkl']}
        self.fileselector_len = {'LiteBird':['c_fid_len.pkl','cov_len.pkl'],
                           'litebird_cmbs4':['c_fid_len_s4.pkl','cov_len_s4.pkl'],
                           'litebird_S4pLB':['c_fid_len_s4.pkl','cov_len_s4.pkl']}        
        
        if basename is None:
            print(f"Likelihood {self.name} requires basename")
            raise AttributeError
        if basename not in list(self.fileselector_len.keys()):
            raise FileNotFoundError
        
        if self.fit_lensed:
            self.fid, self.cov = self.openfile(self.fileselector_len[basename])
        else:
            self.fid, self.cov = self.openfile(self.fileselector_del[basename])
        
        
        
        imin,imax = self.select[0],self.select[-1]+1
        self.cov_inv = np.linalg.inv(self.cov)[imin:imax,imin:imax]
        
        
    
    def openfile(self,files):
        dire = '/global/u2/l/lonappan/workspace/S4bird/Data/'
        fid = pk.load(open(f"{dire}{files[0]}",'rb'))
        cov = pk.load(open(f"{dire}{files[1]}",'rb'))
        return fid, cov
           
    def X(self,cl_th):
        return self.mean/cl_th

    def G(self,cl_th):
        x = self.X(cl_th)
        return np.sign(x-1)* np.sqrt(2*(x - np.log(x) - 1))
    
    def vect(self,theta):
        r,alens = theta
        cl_th = self.cl_theory(r,alens)
        g = self.G(cl_th)
        return g*self.fid
    
    def chi_sq(self,theta):
        vec = self.vect(theta)[self.select]
        l = np.dot(np.dot(vec,self.cov_inv),vec)
        return  l