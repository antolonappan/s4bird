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
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pandas as pd
from tqdm import tqdm 
from getdist import plots, MCSamples
from contextlib import contextmanager
import sys
import matplotlib.ticker as ticker
@contextmanager
def suppress_stdout():
    # Borrowed from
    # https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class cosmology:
    
    def __init__(self,lib_dir,nside,binsize,ini):
        if mpi.rank == 0:
            os.makedirs(lib_dir,exist_ok=True)
            
        self.lib_dir = lib_dir
        self.b = pymaster.NmtBin.from_nside_linear(nside,binsize)
        self.ell = self.b.get_effective_ells()
        self.n_ell = len(self.ell)
        self.dl = self.ell*(self.ell+1)/(2*np.pi)
        self.ini = ini

    def get_spectra(self,r=0):
        fname = os.path.join(self.lib_dir,f"spectra_r{r}.pkl")
        fname2 = f"/global/u2/l/lonappan/workspace/S4bird/Data/spectra_r{r}.pkl"
        if os.path.isfile(fname) or os.path.isfile(fname2):
            try:
                return pk.load(open(fname2,'rb'))
            except:
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

class Delens_Theory:
    def __init__(self,ini,lmax,N0,beam=None,nlevp=0):
        pars = camb.read_ini(ini)
        pars.max_l = lmax
        self.lmax = lmax
        self.results = camb.get_results(pars)
        self.ell = np.arange(self.lmax +1)
        self.n0 = N0
        self.cl_pp = self.results.get_lens_potential_cls()[:,0]
        self.nlevp = np.radians(nlevp/60)**2
        self.fl = np.ones(self.lmax+1) if beam is None else hp.gauss_beam(np.radians(beam/60.),self.lmax,True)[:,2]
        
    @property
    def N0(self):
        N0 = np.zeros(self.lmax+1)
        N0_ = self.n0
        N0[np.arange(len(N0_))] += N0_
        return self.DL*N0
        
    @property    
    def DL(self):
        l = self.ell
        return (l*(l+1))**2 / (2*np.pi)
    @property    
    def dl(self):
        l = self.ell
        return (l*(l+1)) / (2*np.pi)    
    @property
    def cl_pp_res(self):
        return self.cl_pp*(1 - (self.cl_pp/(self.cl_pp+self.N0)))
        
    @property
    def lensed_bb(self):
        bb = np.zeros(self.lmax+1)
        bb_ = self.results.get_lensed_scalar_cls(lmax=self.lmax,CMB_unit='muK')[:,2]
        bb[np.arange(len(bb_))] += bb_
        return (bb/self.dl)
    
    @property
    def tensor_bb(self):
        bb = np.zeros(self.lmax+1)
        bb_ = self.results.get_tensor_cls(lmax=self.lmax,CMB_unit='muK')[:,2]
        bb[np.arange(len(bb_))] += bb_
        return (bb/self.dl)
    
    @property
    def delensed_bb(self):
        bb = np.zeros(self.lmax+1)
        bb_ = self.results.get_lensed_cls_with_spectrum(self.cl_pp_res,lmax=self.lmax,CMB_unit='muK')[:,2]
        bb[np.arange(len(bb_))] += bb_
        return (bb/self.dl)
    @property
    def df_bb(self):
        return self.delensed_bb - self.lensed_bb
    
    def plt_bb(self):
        plt.figure(figsize=(7,7))
        plt.loglog(self.ell,self.lensed_bb,label='Lensed')
        plt.loglog(self.ell,self.delensed_bb,label='Delensed')
        plt.legend(fontsize=20)
        plt.ylim(1e-6,None)
        plt.xlabel('$\ell$',fontsize=20)
        plt.ylabel('$C_\ell^{BB}$',fontsize=20)
        
    def plt_pp(self):
        plt.figure(figsize=(7,7))
        plt.loglog(self.ell,self.cl_pp,label='PP')
        plt.loglog(self.ell,self.cl_pp_res,label='Residual')
        plt.legend(fontsize=20)
        plt.xlabel('$\ell$',fontsize=20)
        plt.ylabel('$C_\ell^{\phi \phi}$',fontsize=20)
        
    #def sigma_r(alens=1,fsky=.5,lmin=10,lmax=150):
    #    f_rr = (ell[lmin:lmax+1] + 0.5)*fsky * ((tensor['bb'][lmin:lmax+1]/((alens*lensed['bb'][lmin:lmax+1]+n)))**2)
    #    return f"{np.sum(f_rr)**-.5:.2e}"


class LH_base:
    
    def __init__(self,lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,fix_alens,cache):
        ini = '/project/projectdirs/litebird/simulations/maps/lensing_project_paper/S4BIRD/CAMB/CAMB.ini'
        self.lib_dir = lib_dir
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
        self.nsamples = nsamples
        
        self.cosmo = cosmology(self.lib_dir,512,10,ini)
        self.tensor = self.cosmo.get_bandpower('T',r=1)
        self.lensing = self.cosmo.get_bandpower_cstm(cl_len)
        self.beam = self.cosmo.get_beam(beam)
        self.noise = self.cosmo.get_noise(nlev_p)
        self.fit_lensed = fit_lensed
        self.fix_alens = fix_alens
        self.cache = cache
        
        
        len_m,len_s,del_m,del_s = eff_lib.get_stat
        self.bias = eff_lib.bias
        bias_s = eff_lib.bias_std
        
        self.select = np.where((self.cosmo.ell >= lmin)& (self.cosmo.ell <= lmax))[0]
        
        self.cov_lib = cov_lib

        if fit_lensed:
            print(f"Fitting Lensed spectra between l={lmin} and l={lmax}")
            self.mean = eff_lib.lib_pcl.get_spectra('lensed',0,99)
        else:
            print(f"Fitting Delensed spectra between l={lmin} and l={lmax}")
            self.mean = eff_lib.lib_pcl.get_spectra('delensed',0,99)
        
        self.name = None
        self.ALENS = None
        
    def chi_sq(self,theta,i):
        pass
    def chi_sq_r(self,theta,i):
        return self.chi_sq([0,theta],i)
    
    def chi_sq_alens(self,theta,i):
        return self.chi_sq([theta,self.ALENS],i)

    def initial_opt(self,i):
        for alens in tqdm(np.linspace(0,1,11)[::-1],desc='Finding Maximum Likelihood Value',unit='guess'):
            #with suppress_stdout():
            swap = False
            if self.fix_alens:
                swap = True
                self.fix_alens = False
                res = opt.minimize(self.chi_sq_r, alens,args=(i))
            else:
                res = opt.minimize(self.chi_sq,[0.0,alens],args=(i))
                
            if np.isnan(res['fun']):
                pass
            else:
                if swap:
                    self.fix_alens = True
                return res
        
    def cl_theory(self,r,Alens):
        th = (r * self.tensor) + (Alens * self.lensing)
        th = th*self.beam**2 + self.noise
        return th

    def log_prior(self,theta):
        if self.fix_alens:
            r = theta
            if  -0.5 < r < 0.5:
                return 0.0
            return -np.inf
        
        else:
            r,Alens= theta
            if  -0.5 < r < 0.5 and 0 < Alens < 1.5:
                return 0.0
            return -np.inf

    def log_probability(self,theta,i):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp  -.5*self.chi_sq(theta,i)

    def posterior(self,i):
        fname_sub = "" if self.fit_lensed else f"_False" 
        fname = os.path.join(self.lib_dir,f"posterior_sim{i}_{self.name}_{self.nsamples}_L{int(self.fit_lensed)}S_{self.select[0]}_{self.select[-1]}_{self.fix_alens}{fname_sub}.pkl")
        if os.path.isfile(fname) and self.cache:
            return pk.load(open(fname,'rb'))
        else:
            res = self.initial_opt(i)['x']
            if self.fit_lensed:
                self.ALENS = res[0]
            else:
                self.ALENS = res[0]
            

            if self.fix_alens:
                print(f"Setting Alens to {self.ALENS}")
                self.fix_alens = False
                res = opt.minimize(self.chi_sq_alens,0.0,args=(i))['x']
                self.fix_alens = True
                    
                pos = np.array([0]) + 1e-4 * np.random.randn(100, 1)
            else:
                pos = np.array(res) + 1e-4 * np.random.randn(100, 2)
                
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,kwargs={'i':i})
            sampler.run_mcmc(pos, self.nsamples,progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pk.dump(flat_samples, open(fname,'wb'),protocol=2)
            return flat_samples

            
    
    def sigma_r(self,i):
        with suppress_stdout():
            samp = MCSamples(samples=self.posterior(i),names=['r'], labels=['r'])
        return f"{float(self.splitter(samp.getInlineLatex('r',limit=1,err_sig_figs=5))[-1]):.2e}"
    
    def sigma_r_old(self,samples):
        r_samp = np.sort(samples)
        r_pos = r_samp[r_samp>0]
        return f"{r_pos[int(len(r_pos)*.683)]:.2e}"
    
    def plot_posterior(self,i,filename=None):
            
        if self.fix_alens:
            labels = ["r"]
        else:
            labels = ["r","Alens"]
        flat_samples = self.posterior(i)
        plt.figure(figsize=(8,8))
        fig = corner.corner(flat_samples, labels=labels,truths=[0] if self.fix_alens else [0,0])
            
    def get_stat(self,i):
        samples = self.posterior(i)
        if self.fix_alens:
            labels = ["r"]
        else:
            labels = ["r","A_{lens}"]
        
        return ParamStat(samples,labels).getTitle()
    
    def sigma_r_array(self):
        r = []
        for i in range(100):
            r.append(self.sigma_r(i))
        return np.array(r).astype(float)
    
    def plot_hist(self,savefig=False):
        fname = os.path.join(self.lib_dir,f"L{int(self.fit_lensed)}.png")
        r = self.sigma_r_array()
        plt.figure(figsize=(8,8))
        sns.distplot(r,hist=True,kde=True,bins=20)
        pdf, pr = np.histogram(r,bins=50)
        s = np.where(pdf == np.max(pdf))[0][0]
        plt.axvline(pr[s],label=f"Mode = {pr[s]:.2e}",c='k')
        plt.axvline(np.mean(r), label=f"Mean = {np.mean(r):.2e}",c='b')
        plt.axvline(np.median(r), label=f"Median = {np.median(r):.2e}",c='r')
        plt.legend(fontsize=20)
        if savefig:
            plt.savefig(fname,bbox_inches='tight')
            
    def splitter(self,chuma):
        first = chuma.split('=')[-1]
        second = first.split('\pm')
        if len(second) < 2:
            second = second[0].split('}_{-')[0].split('^{+')
        return second
            
    
    def plot_stat(self,bw=1,bins=10,savefig=False):
        if self.fix_alens:
            to_remove = []
            for i in range(100):
                try:
                    NULL = self.sigma_r(i)
                except:
                    to_remove.append(i)
            print(f"Bad posteriors: {len(to_remove)}")
            fname = os.path.join(self.lib_dir,f"Stat_FA_L{int(self.fit_lensed)}.png")
            fname2 = os.path.join(self.lib_dir,f"Stat_FA_L{int(self.fit_lensed)}.pkl")
            name = ['r']
            label = ['r']
            samples = []
            if not os.path.isfile(fname2):
                r,sigma_r = [],[]
                for i in tqdm(range(100),desc='Plotting posteriors',unit='simulations'):
                    if i in to_remove:
                        pass
                    else:
                        with suppress_stdout():
                            try:
                                samps = MCSamples(samples=self.posterior(i),names = name, labels = label)
                                samples.append(samps)
                                r_text = self.splitter(samps.getInlineLatex('r',limit=1,err_sig_figs=5))
                                r.append(float(r_text[0]))
                                sigma_r.append(float(r_text[-1]))
                            except:
                                print(i)

                pk.dump((r,sigma_r),open(fname2,'wb'))
            else:
                r,sigma_r = pk.load(open(fname2,'rb'))
            
            fig, axs = plt.subplots(2, 1, figsize=(5, 10))
            #sns.kdeplot(sigma_r, ax=axs[0], bw=bw,fill=True)
            axs[0].hist(sigma_r,bins=10)
            axs[0].axvline(np.mean(sigma_r),label=f"Mean = {np.mean(sigma_r):.2e}",c='k',ls=':')
            axs[0].tick_params(labelrotation=45,labelsize=15)
            axs[0].set_xlabel("$\sigma_r$",fontsize=15)
            axs[0].legend(loc="lower left",fontsize=15)


            sr_m = float(self.sigma_r_old(r))
            #sns.kdeplot(r, ax=axs[1], bw=bw,fill=True)
            axs[1].hist(r,bins=10)
            #axs[1].axvline(sr_m,label=f"68% CL = {sr_m:.2e}",c='k',ls=':')
            #axs[1].axvline(-sr_m,c='k',ls=':')
            axs[1].tick_params(labelsize=15,labelrotation=45)
            axs[1].set_xlabel('$r$',fontsize=15)
            #axs[1].legend(loc="lower left",fontsize=15)
            plt.subplots_adjust(hspace=.4)
            if savefig:
                plt.savefig(fname,bbox_inches='tight')

            
        else:
            fname = os.path.join(self.lib_dir,f"Stat_L{int(self.fit_lensed)}.png")
            fname2 = os.path.join(self.lib_dir,f"Stat_L{int(self.fit_lensed)}.pkl")
            name = ['r','alens']
            label = ['r','A_{lens}']
            samples = []
            if not os.path.isfile(fname2):
                r,alens,sigma_r = [],[],[]
                for i in tqdm(range(100),desc='Plotting posteriors',unit='simulations'):
                    with suppress_stdout():
                        samps = MCSamples(samples=self.posterior(i),names = name, labels = label)
                        samples.append(samps)
                        r_text = self.splitter(samps.getInlineLatex('r',limit=1,err_sig_figs=5))
                        a_text = self.splitter(samps.getInlineLatex('alens',limit=1,err_sig_figs=5))

                        r.append(float(r_text[0]))
                        alens.append(float(a_text[0]))
                        sigma_r.append(float(r_text[-1]))

                pk.dump((r,alens,sigma_r),open(fname2,'wb'))
            else:
                r,alens,sigma_r = pk.load(open(fname2,'rb'))

            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            axs[1,0].hist(r,bins=bins,density=True,label=f"Mean = {np.mean(r):.2e}")
            axs[1,0].set_xlabel('$r$')
            axs[1,0].legend()
            axs[0,1].hist(alens,bins=bins,density=True,label=f"Mean = {np.mean(alens):.2f}")
            axs[0,1].set_xlabel('$A_{lens}$')
            axs[0,1].legend()
            axs[0,0].hist(sigma_r,bins=bins,density=True,label=f"Mean = {np.mean(sigma_r):.2e}")
            axs[0,0].set_xlabel("$\sigma_r$")
            axs[0,0].legend()
            axs[1,1].hist2d(r,alens)
            axs[1,1].set_xlabel('$r$')
            axs[1,1].set_ylabel('$A_{lens}$')
            if savefig:
                plt.savefig(fname,bbox_inches='tight')
        
        
    
        
class LH_simple(LH_base):
    def __init__(self,lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,basename,fix_alens,cache):
        super().__init__(lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,fix_alens,cache)
        self.name = 'simple'
        
        if self.fit_lensed:
            self.fid = self.cov_lib.lensed_fid
            cov = self.cov_lib.lensed_fid_cov
        else:
            self.fid = self.cov_lib.delensed_fid
            cov = self.cov_lib.delensed_fid_cov
        self.cov = np.zeros(cov.shape)
        np.fill_diagonal(self.cov, np.diag(cov))
        self.cov_inv = np.linalg.inv(cov) 
    
    def cl_theory(self,r,alens):
        th = (r * self.tensor) + alens* self.lensing
        th = th[self.select]*self.beam[self.select]**2  + self.noise[self.select]
        return th
    def log_prior(self,theta):
        r,alens = theta
        if  -0.1 < r < 0.1 and 0 < alens <1.5:
            return 0.0
        return -np.inf
    
    def vect(self,theta):
        r, alens= theta
        cl_th = self.cl_theory(r, alens)    
        return cl_th - self.fid
    
    def log_probability(self,theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp  -.5*self.chi_sq(theta)
    
    def posterior(self):
        fname = os.path.join(self.lib_dir,f"posterior_sim.pkl")
        if os.path.isfile(fname) and self.cache:
            return pk.load(open(fname,'rb'))
        else:
            pos = np.array([0,.5]) + 1e-4 * np.random.randn(100, 2)

                
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,)
            sampler.run_mcmc(pos, self.nsamples,progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pk.dump(flat_samples, open(fname,'wb'),protocol=2)
            return flat_samples
        
    def chi_sq(self,theta):
        vec = self.vect(theta)
        l = np.dot(np.dot(vec,self.cov_inv),vec)
        return  l
    def plot_posterior(self):
        labels = ["r","alens"]
        flat_samples = self.posterior()
        plt.figure(figsize=(8,8))
        fig = corner.corner(flat_samples, labels=labels,truths=[0,0])
        
    def sigma_r(self):
        with suppress_stdout():
            samp = MCSamples(samples=self.posterior(),names=['r',"alens"], labels=['r',"alens"])
        return f"{float(self.splitter(samp.getInlineLatex('r',limit=1,err_sig_figs=5))[-1]):.2e}"


class LH_HL(LH_base):
    def __init__(self,lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,basename,fix_alens,cache):
        super().__init__(lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,fix_alens,cache)
        self.name = 'HL'

        
        
        if self.fit_lensed:
            self.fid = self.cov_lib.lensed_fid
            cov = self.cov_lib.lensed_fid_cov
        else:
            self.fid = self.cov_lib.delensed_fid
            cov = self.cov_lib.delensed_fid_cov
        
        self.cov = np.zeros(cov.shape)
        np.fill_diagonal(self.cov, np.diag(cov))

        self.cov_inv = np.linalg.inv(self.cov)
           
    def X(self,cl_th,i):
        if self.fit_lensed:
            return self.mean[i]/cl_th
        else:
            return (self.mean[i]-self.bias)/cl_th

    def G(self,cl_th,i):
        x = self.X(cl_th,i)
        return np.sign(x-1)* np.sqrt(2*(x - np.log(x) - 1))
    
    def vect(self,theta,i):
        if self.fix_alens:
            r = theta
            cl_th = self.cl_theory(r,self.ALENS)
        else:
            r,alens = theta
            cl_th = self.cl_theory(r,alens)            
        g = self.G(cl_th,i)
        return g[self.select]*self.fid
    
    def chi_sq(self,theta,i):
        vec = self.vect(theta,i)
        l = np.dot(np.dot(vec,self.cov_inv),vec)
        return  l
    

class LH_HL_mod(LH_HL):
    def __init__(self,lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,basename,fix_alens,cache):
        super().__init__(lib_dir,eff_lib,cov_lib,nsamples,cl_len,nlev_p,beam,lmin,lmax,fit_lensed,basename,fix_alens,cache)
        self.fix_alens = True
        self.name = 'HL_mod'
        
    def X(self,cl_th,i):
        if self.fit_lensed:
            return self.mean[i][self.select]/cl_th
             
        else:
            return (self.mean[i]-self.bias)[self.select]/cl_th
        
    def vect(self,theta,i):
        if self.fix_alens:
            r = theta
            cl_th = self.cl_theory(r,self.ALENS)
        else:
            r,alens = theta
            cl_th = self.cl_theory(r,alens)            
        g = self.G(cl_th,i)
        return g*self.fid
    
    def cl_theory(self,r,Alens=None):
        th = r * self.tensor
        th = th*self.beam**2 
        return th[self.select] +self.fid
    
    def initial_opt(self,i):
        res = opt.minimize(self.chi_sq,[0.0],args=(i))
        return res
    
    def posterior(self,i):
        fname = os.path.join(self.lib_dir,f"posterior_sim{i}_{self.name}_{self.nsamples}_L{int(self.fit_lensed)}S_{self.select[0]}_{self.select[-1]}.pkl")
        if os.path.isfile(fname) and self.cache:
            return pk.load(open(fname,'rb'))
        else:
            pos = np.array([0]) + 1e-4 * np.random.randn(100, 1)
                
            nwalkers, ndim = pos.shape
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,kwargs={'i':i})
            sampler.run_mcmc(pos, self.nsamples,progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pk.dump(flat_samples, open(fname,'wb'),protocol=2)
            return flat_samples
    def plot_spectra(self,i):
        if self.fit_lensed:
            plt.loglog(self.mean[i][self.select],label='data')
        else:
            plt.loglog((self.mean[i]-self.bias)[self.select],label='data')
        plt.loglog(self.cl_theory(0),label='theory')
        plt.legend()
        
    def sigma_r(self,i):
        with suppress_stdout():
            samp = MCSamples(samples=self.posterior(i),names=['r'], labels=['r'])
        return f"{float(self.splitter(samp.getInlineLatex('r',limit=1,err_sig_figs=5))[-1]):.2e}"
        

        
class ParamStat:
    
    def __init__(self,chains,label,cl=0.682,paded=True,smooth=True):
        self.chains = chains
        self.cl = cl
        self.labels = label
        self.paded = paded
        self.smooth = smooth
        
    def __get_hist(self,data):
        hist, edges = np.histogram(data,bins=10000,density=True)
        edge_centers = 0.5 * (edges[1:]+edges[:-1])

        xs = np.linspace(edge_centers[0], edge_centers[-1], 10000)
        ys = interp1d(edge_centers, hist, kind="linear")(xs)

        if self.smooth:
            ys = gaussian_filter(ys,10)

        if self.paded:
            #padding the data for cut chains
            n_pad = 1000
            x_start = xs[0] * np.ones(n_pad)
            x_end = xs[-1] * np.ones(n_pad)
            y_start = np.linspace(0, ys[0], n_pad)
            y_end = np.linspace(ys[-1], 0, n_pad)
            xs = np.concatenate((x_start, xs, x_end))
            ys = np.concatenate((y_start, ys, y_end))

        return xs, ys

    def __get_cumlative_pdf(self, data):
        _, ys = self.__get_hist(data,)
        cs = ys.cumsum()
        cs = cs / cs.max()
        return cs
    
    def get_stat(self,data):

        xs, ys = self.__get_hist(data)
        cs = self.__get_cumlative_pdf(data)
        startIndex = ys.argmax()
        maxVal = ys[startIndex]
        minVal = 0
        threshold = 0.03
        x1 = None
        x2 = None
        count = 0
        values = []
        while x1 is None:
            mid = (maxVal + minVal) / 2.0
            count += 1

            if count > 50:
                raise ValueError("Failed to converge")
            i1 = startIndex - np.where(ys[:startIndex][::-1] < mid)[0][0]
            i2 = startIndex + np.where(ys[startIndex:] < mid)[0][0]
            values.append((cs[i2],cs[i1]))
            area = cs[i2] - cs[i1]
            deviation = np.abs(area - self.cl)
            if deviation < threshold:
                x1 = xs[i1]
                x2 = xs[i2]
            elif area < self.cl:
                maxVal = mid
            elif area > self.cl:
                minVal = mid

        res = [x1, xs[startIndex], x2]


        lower, maximum, upper =res

        upper_error = upper - maximum
        lower_error = maximum - lower

        return upper_error, lower_error, xs[startIndex]

    def __get_tex(self,upper,lower,ml):
        if f"{upper:.2f}" != f"{lower:.2f}":
            return f"{ml:.2f}^{{+{upper:.2e}}}_{{-{lower:.2e}}}"
        else:
            return f"{ml:.2f} \pm {upper:.2e}"

    def getInlinetex(self,data):
        upper,lower, ml =self.get_stat(data)
        return self.__get_tex(upper,lower,ml)
    
    def getTitle(self):
        titles = []
        for i in range(len(self.labels)):
            data = self.chains[:,i]
            if i == 0:
                data = np.sort(data)
                data = data[data>0]
            titles.append(f"${self.labels[i]} = {self.getInlinetex(data)}$" )
        return titles
    
class LH_base2:
    
    def __init__(self,cov,nsample,theory,b,lmin,lmax,which,use_diag):
        self.nsamples = nsample
        self.theory = theory
        self.lensed_bb = self.theory.lensed_bb
        self.delensed_bb = self.theory.delensed_bb
        self.tensor_bb = self.theory.tensor_bb
        self.b = b
        ell = b.get_effective_ells()
        self.select = np.where((ell>lmin) & (ell<lmax))[0]
        self.ell = ell[self.select]
        self.which = which
        
        if use_diag:
            cov_ = np.zeros(cov.shape)
            np.fill_diagonal(cov_, np.diag(cov))
        else:
            cov_ = cov
        self.cov = cov_
        self.cov_inv = np.linalg.inv(cov_) 
    
    def chi_sq(self):
        pass
    
    def cl_theory_lensed(self,r):
        th = (r * self.tensor_bb) + self.lensed_bb
        th = th*self.theory.fl**2 + self.theory.nlevp
        return self.b.bin_cell(th[:self.b.lmax+1])[self.select]
    
    def cl_theory_delensed(self,r):
        th = (r * self.tensor_bb) + self.delensed_bb
        th = th*self.theory.fl**2 + self.theory.nlevp
        return self.b.bin_cell(th[:self.b.lmax+1])[self.select]
    
    def cl_theory(self,r):
        if self.which == 'lensed':
            return self.cl_theory_lensed(r)
        elif  self.which == 'delensed':
            return self.cl_theory_delensed(r)
        else:
            pass
        
        
    def log_prior(self,theta):
        r= theta
        if  -0.5 < r < 0.5:
            return 0.0
        return -np.inf

    def log_probability(self,theta,i):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp  -.5*self.chi_sq(theta,i)
    

    def posterior(self,i):
        pos = np.array([0]) + 1e-4 * np.random.randn(100, 1)
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability,kwargs={'i':i})
        sampler.run_mcmc(pos, self.nsamples,progress=True)
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        return flat_samples
    
    def report(self,i):
        samples = i
        cut_samples = samples[samples > 0]
        with suppress_stdout():
            cut_samp = MCSamples(samples=cut_samples,names=['r'], labels=['r'],ranges={'r':(0, None)})
            return cut_samp.getInlineLatex('r',limit=1,err_sig_figs=5)

    def sigma_r(self,i):
        samples = self.posterior(i)
        cut_samples = samples[samples > 0]
        with suppress_stdout():
            cut_samp = MCSamples(samples=cut_samples,names=['r'], labels=['r'],ranges={'r':(0, None)})
            return cut_samp.getInlineLatex('r',limit=1,err_sig_figs=5)   
        

        
    def plot_spectra(self,i):
        plt.loglog(self.ell,self.cl_theory_lensed(0),label='theoryL')
        plt.loglog(self.ell,self.cl_theory_delensed(0),label='theoryD')
        plt.errorbar(self.ell,i[self.select],yerr=np.sqrt(np.diag(self.cov)),label='spectra')
        plt.legend()
    
    def plot_posterior(self,i):
        labels = ["r"]
        flat_samples = self.posterior(i)
        print(self.report(flat_samples))
        plt.figure(figsize=(8,8))
        fig = corner.corner(flat_samples, labels=labels,truths=[0] )
        
class LH_simple2(LH_base2):

    def __init__(self,cov,nsample,theory,b,lmin=10,lmax=100,which='delensed',use_diag=False):
        super().__init__(cov,nsample,theory,b,lmin,lmax,which,use_diag)
    
    def vect(self,theta,i):
        r = theta
        cl_th = self.cl_theory(r) 

        return cl_th - i[self.select]
    
    def chi_sq(self,theta,i):
        vec = self.vect(theta,i)
        l = np.dot(np.dot(vec,self.cov_inv),vec)
        return  l
    
class LH_HL2(LH_base2):

    def __init__(self,c,cov,nsample,theory,b,lmin=10,lmax=100,which='delensed',use_diag=False):
        super().__init__(cov,nsample,theory,b,lmin,lmax,which,use_diag)
        self.c = c[self.select]

    def G(self,cl_th,i):
        x = i[self.select]/cl_th
        return np.sign(x-1)* np.sqrt(2*(x - np.log(x) - 1))
    
    def vect(self,theta,i):
        r = theta
        cl_th = self.cl_theory(r)
        g = self.G(cl_th,i)
        return g*self.c
    
    def chi_sq(self,theta,i):
        vec = self.vect(theta,i)
        l = np.dot(np.dot(vec,self.cov_inv),vec)
        return  l