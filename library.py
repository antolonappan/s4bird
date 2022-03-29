import lenspyx
import hashlib
import numpy as np
import healpy as hp
import os
import corner
import emcee
import pymaster as nmt
import _pickle as plk
import matplotlib.pyplot as pl
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.optimize as opt
from tqdm import tqdm 
import pickle as pk
from plancklens import utils
from plancklens.helpers import mpi
from shutil import copyfile


class Delensing:
    __slots__ = ["lib_dir","sims","ivfs","qlms","qresp","nhl","n_sim","lmax_qlm","cl_ppf",
                 "nside","maskpath","save_template","verbose","qnorm","transf",'key',
                 "first_set","last_set","mid_sim","of_by"]
    
    def __init__(self,lib_dir,
                 sims,
                 ivfs,
                 qlms,
                 qresp,
                 nhl,
                 n_sim,
                 lmax_qlm,
                 cl_ppf,
                 nside,
                 maskpath,
                 key,
                 of_by,
                 transf=None,
                 save_template=False,
                 verbose=False):
        self.key =  key
        print(f"Delensing uses QE: {self.key}")
        self.lib_dir = lib_dir
        self.sims = sims
        self.ivfs = ivfs
        self.qlms = qlms
        self.qresp = qresp.get_response(self.key, 'p')
        self.nhl = nhl
        self.n_sim = n_sim
        self.lmax_qlm = lmax_qlm
        self.cl_ppf = cl_ppf
        self.nside = nside
        self.maskpath = maskpath
        self.save_template = save_template
        self.verbose = verbose
        self.transf = transf
        self.of_by = of_by
        
        #For Mean Field subtraction
        self.mid_sim = int(self.n_sim/2)
        first_set = np.arange(0,self.mid_sim/2, dtype=int)
        last_set = np.arange(self.n_sim - (self.mid_sim/2), self.n_sim, dtype=int)
        self.first_set = self.qlms.get_sim_qlm_mf(self.key,first_set)
        self.last_set = self.qlms.get_sim_qlm_mf(self.key,last_set)
        
        self.qnorm = utils.cli(self.qresp)
        
        fnhash = os.path.join(self.lib_dir, "delens_sim_hash.pk")
        if (mpi.rank == 0) and (not os.path.exists(fnhash)):
            if not os.path.exists(self.lib_dir): os.makedirs(self.lib_dir)
            pk.dump(self.hashdict(), open(fnhash, 'wb'), protocol=2)
        mpi.barrier()
        
        #utils.hash_check(pk.load(open(fnhash, 'rb')), self.hashdict())

    
    def hashdict(self):
        return {'ivfs':self.ivfs.hashdict(),
                'qlms':self.qlms.hashdict(),
                'n_sim':self.n_sim,
                'lmax_qlm': self.lmax_qlm,
                'nside': self.nside
               }

    
    def mf_corrected_qlm(self,idx):
        qlm = self.qlms.get_sim_qlm( self.key, idx)
        qlm_mf = self.first_set if idx > self.mid_sim else self.last_set
        return qlm - qlm_mf
    
    def get_N0(self,idx):
        nhl = self.nhl.get_sim_nhl(idx,  self.key,  self.key)
        return nhl*self.qnorm**2
    
    def plot_clpp(self,idx):
        ell = np.arange(0, self.lmax_qlm)
        nhl = self.nhl.get_sim_nhl(idx,  self.key,  self.key)
        w = lambda ell : ell ** 2 * (ell + 1.) ** 2 * 0.5 / np.pi
        w2 = lambda ell : ell*(ell+1) * 0.5 /np.pi
        qlm = self.mf_corrected_qlm(idx)
        clpp = hp.alm2cl(qlm)[ell] * self.qnorm[ell] ** 2 / self.qlms.fsky12 * w(ell)
        noise = nhl[ell] * self.qnorm[ell] ** 2 * w(ell)
        pl.figure(figsize=(8,8))
        pl.loglog(ell,(clpp-noise),label=r'$C_L^{\phi\phi, \rm recon}$')
        pl.loglog(ell,noise,label=r'$N_L^0$')
        pl.loglog(ell, hp.alm2cl(self.qlm_wf(idx))[ell] *w(ell), label=r'$C_L^{\phi\phi, \rm WF\: reco}$')
        pl.loglog(ell, self.cl_ppf[ell] *  w(ell), c='k', label=r'$C_L^{\phi\phi, \rm fid}$')
        pl.xlabel('$L$')
        pl.ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$')
        pl.legend(fontsize=20)
    
    def get_fl(self,idx):
        ell = np.arange(0, self.lmax_qlm)
        nhl = self.nhl.get_sim_nhl(idx,  self.key,  self.key)
        fl = self.cl_ppf[ell]/(self.cl_ppf[ell]+(nhl[ell] * self.qnorm[ell] ** 2))
        fl[0] = 0
        fl[1] = 0
        return fl   
    
    def qlm_wf(self,idx):
        qlm_norm = hp.almxfl(self.mf_corrected_qlm(idx),self.qnorm)
        fl = self.get_fl(idx)
        return hp.almxfl(qlm_norm,fl)
    
    
    def kappa_wf(self,idx):
        wplm = self.qlm_wf(idx)
        walpha = hp.almxfl(wplm, np.sqrt(np.arange(self.lmax_qlm + 1, dtype=float) * np.arange(1, self.lmax_qlm + 2)))
        ftl = np.ones(self.lmax_qlm + 1, dtype=float) * (np.arange(self.lmax_qlm + 1) >= 10)
        walpha = hp.almxfl(walpha,ftl)
        return walpha
    
    def get_template(self,idx):
        filename = [os.path.join(self.lib_dir,f"sim_template_{self.of_by}_{self.key}_{idx:04d}_Q.fits"),
                    os.path.join(self.lib_dir,f"sim_template_{self.of_by}_{self.key}_{idx:04d}_U.fits")]
        if os.path.isfile(filename[0]) and os.path.isfile(filename[1]):
            return hp.read_map(filename[0],verbose=self.verbose), hp.read_map(filename[1],verbose=self.verbose)
        
        else:
            elm_wf = self.ivfs.get_sim_emliklm(idx)
        
        
            Q, U  = lenspyx.alm2lenmap_spin([elm_wf, None], [self.kappa_wf(idx), None], self.nside, 2, facres=-1,verbose=self.verbose)
            del elm_wf
            Q = hp.smoothing(Q,beam_window=self.transf)
            U = hp.smoothing(U,beam_window=self.transf)
            print("Transfer function applied to the Template")
            
            if self.save_template:
                hp.write_map(filename[0], Q)
                hp.write_map(filename[1], U)
            return Q, U
        
    def get_lensed_field(self,idx):
        return self.sims.get_sim_pmap(idx)
    
    def get_delensed_field(self,idx):
        filename = [os.path.join(self.lib_dir,f"sim_delens_{self.of_by}_{self.key}_{idx:04d}_Q.fits"),
                    os.path.join(self.lib_dir,f"sim_delens_{self.of_by}_{self.key}_{idx:04d}_U.fits")]
        if os.path.isfile(filename[0]) and os.path.isfile(filename[1]):
            return hp.read_map(filename[0]), hp.read_map(filename[1])
        
        else:
            Q, U = self.get_template(idx)
            Q_d, U_d = self.get_lensed_field(idx) 
                
            
            Q_t = Q_d - Q
            U_t = U_d - U
            hp.write_map(filename[0], Q_t)
            hp.write_map(filename[1], U_t)
            del(Q,U,Q_d,U_d)
            
            return Q_t, U_t
    
    def missing_idx(self,temp=False):
        which = 'delens' if not temp else 'template'
        miss_u = []
        miss_q = []
        for i in range(self.n_sim):
            qname = os.path.join(self.lib_dir,f"sim_{which}_{self.key}_{i:04d}_Q.fits")
            uname = os.path.join(self.lib_dir,f"sim_{which}_{self.key}_{i:04d}_U.fits")
            if not os.path.isfile(qname):
                miss_q.append(i)
            if not os.path.isfile(uname):
                miss_u.append(i)
        return set(miss_u + miss_q)
            
class Efficency:
    __slots__ = ["lib_dir","lib_pcl","n_sim","fiducial","bias_do","bias","bias_std","key"]
    def __init__(self,lib_dir,lib_pcl,n_sim,fiducial,bias_do=False,bias_file=None):
        self.lib_dir = lib_dir
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
        
        self.lib_pcl = lib_pcl
        self.key = self.lib_pcl.delens_lib.key
        self.n_sim = n_sim
        self.fiducial = fiducial
        self.bias_do = bias_do
        self.bias = None
        self.bias_std = None
        if bias_do:
            assert bias_file != None
            self.bias, self.bias_std = pk.load(open(bias_file,'rb'))
            
            
    def save_bias(self):
        fname = os.path.join(self.lib_dir,f"bias_{self.key}.pkl")
        if os.path.isfile(fname):
            print(f"Bias file already exist at {fname}. This file will be overwritted!")
        bias = []
        for i in tqdm(range(self.n_sim),desc="Calculating Bias", unit='simulation'):
            bias.append(self.lib_pcl.get_delensed_cl(i)[3]-self.lib_pcl.get_lensed_cl(i)[3])
        bias = np.array(bias)
        bias = (np.mean(bias,axis=0),np.std(bias,axis=0))
        pk.dump(bias, open(fname,'wb'),protocol=2)
        print(f"Bias saved to :{fname}")
            
    
    @property
    def get_stat(self):
        len_spectra = []
        del_spectra = []
        for i in tqdm(range(self.n_sim), desc='Mean and STD of Bandpower', unit='simulation'):
            len_spectra.append(self.lib_pcl.get_lensed_cl(i)[3])
            del_spectra.append(self.lib_pcl.get_delensed_cl(i)[3])
        len_spectra, del_spectra = np.array(len_spectra), np.array(del_spectra)
        return (np.mean(len_spectra,axis=0),
                np.std(len_spectra,axis=0),
                np.mean(del_spectra,axis=0),
                np.std(del_spectra,axis=0))
    
    def plot_stat(self,lmax=50,ymin=None,ymax=10e1,savefig=False,debias=True,filename='Eff_stat.png'):
        
    
        fwhm = hp.gauss_beam(np.radians(30/60.),len(self.fiducial)-1,True)[:,2]

        cl2dl = self.lib_pcl.cl2dl
        l = np.arange(len(self.fiducial))
        n = np.ones(len(l))*np.radians(2.15/60)**2
        dl = l*(l+1)/(2*np.pi)
        len_m, len_s, del_m, del_s = self.get_stat
        pl.figure(figsize=(8,8))
        pl.errorbar(self.lib_pcl.ell,  cl2dl*len_m, yerr=cl2dl*len_s, fmt='r.', markersize='10', label='Lensed')
        pl.errorbar(self.lib_pcl.ell,  cl2dl*del_m, yerr=cl2dl*del_s, fmt='g.', markersize='10',label='Delensed-biased')
        if debias:
            pl.errorbar(self.lib_pcl.ell,  cl2dl*(del_m-self.bias), yerr=cl2dl*np.sqrt(del_s**2 + self.bias_std**2), fmt='b.', markersize='10',label='Delensed-debiased')
        pl.plot(l, ((self.fiducial*fwhm**2) +n)*dl, label='Fiducial')
        pl.xscale('log')
        pl.yscale('log')
        pl.xlabel('$\ell$', fontsize='20')
        pl.ylabel('$D_\ell^{BB}$', fontsize='20')
        pl.xticks(fontsize='20')
        pl.yticks(fontsize='20')
        pl.legend(fontsize='20')
        pl.grid()
        pl.xlim(10,lmax)
        pl.ylim(ymin,ymax)
        if savefig:
            pl.savefig(filename,bbox_incehs='tight')
    
    @property
    def spectrum_difference(self):
        if self.bias_do:
            fname = os.path.join(self.lib_dir,f"spectra_diff_w_bias_{self.key}.pkl")
        else:
            fname = os.path.join(self.lib_dir,f"spectra_diff_wo_bias_{self.key}.pkl")
    
        if os.path.isfile(fname):
            l,diff,std =  pk.load(open(fname,'rb'))
            return diff,std
        len_m, len_s, del_m, del_s = self.get_stat
        diff = del_m - len_m   
        if self.bias_do:
            print('Bias subtracted')
            diff = diff - self.bias
            std = np.sqrt(len_s**2 + del_s**2 + self.bias_std**2)
        else:
            std = np.sqrt(len_s**2 + del_s**2)
            print('Bias not subtracting')
        
        pk.dump([self.lib_pcl.ell,diff,std], open(fname,'wb'))
        return diff,std
    
    
    def plot_spectrum_difference(self,lmax=500,ymin=-3,ymax=1,savefig=False,filename='spectra_difference.png'):
        mean,std = self.spectrum_difference 
        def get_beam(beam):
            b = nmt.NmtBin.from_nside_linear(512,10)
            fwhm = hp.gauss_beam(np.radians(beam/60.),b.lmax,True)[:,2]
            return b.bin_cell(fwhm)

        ref = self.lib_pcl.b.bin_cell(self.fiducial[:self.lib_pcl.b.lmax+1])
        pl.figure(figsize=(8,8))
        pl.errorbar(self.lib_pcl.ell,mean*10**6, yerr=std*10**6, fmt='o')
        pl.plot(self.lib_pcl.ell,-ref*10**6,label='$-C_\ell^{BB}$',c='r')
        pl.plot(np.zeros(lmax+1), c='k',linestyle=':')
        pl.ylim(ymin,ymax)
        pl.xlim(2,lmax)
        pl.legend(fontsize='15')
        pl.xlabel('$\ell$',fontsize='20')    
        pl.ylabel("$10^6(C_\ell^{del} - C_\ell^{data})$",fontsize='20')
        if savefig:
            pl.savefig(filename,bbox_inces='tight')
        
        
    def fit_efficency(self,lmin=20,lmax=200):
        mean,std = self.spectrum_difference
        x = self.lib_pcl.ell
        var = (std)**2
        
        def get_beam(beam):
            b = nmt.NmtBin.from_nside_linear(512,10)
            fwhm = hp.gauss_beam(np.radians(beam/60.),b.lmax,True)[:,2]
            return b.bin_cell(fwhm)
        """
        ell = np.arange(0, len(self.fiducial))
        f2 = interp1d(ell, self.fiducial[ell],kind='linear')
        
        sel = np.where((x >= lmin)& (x <= lmax))[0]
        def chi_sq(epsilon):
            num = mean[sel] - (epsilon * (0-f2(x[sel])) )
            return np.sum((num*10**6)**2 / var[sel])
        """
        ref = self.lib_pcl.b.bin_cell(self.fiducial[:self.lib_pcl.b.lmax+1])#*get_beam(30)
        sel = np.where((x >= lmin)& (x <= lmax))[0]
        def chi_sq(epsilon):
            num = mean[sel] - (epsilon * (0-ref[sel]) )
            return np.sum((num)**2 / var[sel])
        
        x0 = [.5]
        res = opt.minimize(chi_sq, x0)
        
        print(f"OPTIMISATION INFO: {res['message']}")
        print(f"OPTIMISATION INFO: Chisq_{len(sel)} = {res['fun']}")
        print(f"Efficency btw l={lmin} and l={lmax}: {res['x'][0]*100}")
        
        return res['x'][0]
    
       


class Pseudo_cl:
    
    def __init__(self,lib_dir,delens_lib,maskpre,fwhm,purify_b,nside,fsky,binsize,apo_scale,apo_method,of_by):
        
        self.delens_lib = delens_lib
        self.nside = nside
        self.fwhm = fwhm
        self.purify_b = bool(purify_b)
        self.fsky = str(fsky).replace('.','p')
        self.binsize = binsize
        self.apo_scale = int(apo_scale)
        self.apo_method = apo_method
        self.b = nmt.NmtBin.from_nside_linear(self.nside, self.binsize)
        if self.fwhm == 'None':
            self.beam = None
        else: 
            self.beam = hp.gauss_beam(np.radians(float(self.fwhm)/60),lmax=self.b.lmax)
        
        self.of_by = of_by
        
        self.maskpath = f"{maskpre}_Nside{nside}_fsky_{self.fsky}_Apo_{apo_method}_Deg_{int(apo_scale)}.fits"
        self.mask = hp.read_map(self.maskpath,dtype=np.float64)

        # PATHS
        self.lib_dir = lib_dir
        self.workspace_dir = os.path.join(lib_dir,'workspace')
        self.cls_dir = os.path.join(lib_dir,'C_l')
        if mpi.rank == 0:
            for path in [self.lib_dir,self.workspace_dir,self.cls_dir]:
                os.makedirs(path,exist_ok=True)
        mpi.barrier()
        
        #WORKSPACE INITIALISATION
        self.workspace = nmt.NmtWorkspace()
        self.get_coupling_matrix
        

        self.ell = self.b.get_effective_ells()
        self.cl2dl = self.ell * (self.ell + 1)/ (2 * np.pi)
        
    
    @property
    def get_coupling_matrix(self):
        fname = os.path.join(self.workspace_dir, f"coupling_matrix_Nside{self.nside}_fsky_{self.fsky}_beam_{self.fwhm}_Apo_{self.apo_method}_Deg_{self.apo_scale}.fits")
        if not os.path.isfile(fname):
            print("Computing coupling Matrix")
            mask_f = nmt.NmtField(self.mask,[self.mask,self.mask],purify_b=self.purify_b,beam=self.beam)
            self.workspace.compute_coupling_matrix(mask_f, mask_f, self.b)
            del mask_f
            if mpi.rank == 0:
                self.workspace.write_to(fname)
                print(f"Coupling Matrix saved to {fname}")
            mpi.barrier()
        else:
            print(f"Reading coupling matrix from {fname}")
            self.workspace.read_from(fname)
        
        
    
        
    def get_cls(self,idx,which):
        if which == 'lensed':
            filename = os.path.join(self.cls_dir,f"sims_{which}_fsky_{self.fsky}_beam_{self.fwhm}_Apo_{self.apo_method}_Deg_{self.apo_scale}_{idx:04d}.pkl")
        else:
            filename = os.path.join(self.cls_dir,f"sims_{self.of_by}_{which}_fsky_{self.fsky}_beam_{self.fwhm}_Apo_{self.apo_method}_Deg_{self.apo_scale}_{self.delens_lib.key}_{idx:04d}.pkl")
        if os.path.isfile(filename):
            return pk.load(open(filename, 'rb'))
        else:
            if which == 'lensed':
                Q, U = self.delens_lib.get_lensed_field(idx)
            elif which == 'delensed':
                Q, U = self.delens_lib.get_delensed_field(idx)
            elif which == 'template':
                Q, U = self.delens_lib.get_template(idx)
            else:
                raise ValueError
            
            print('computing fields')
            print(f"maps downgraded to NSIDE {self.nside}")
            field = nmt.NmtField(self.mask, [hp.ud_grade(Q,self.nside),hp.ud_grade(U,self.nside)],purify_b=self.purify_b,beam=self.beam)
            del(Q,U)
            print('computing cls')
            clss = self.workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[3] #only save BB spectra
            del field
            pk.dump(clss, open(filename, 'wb'), protocol=2)             
            return clss
        
    def get_lensed_cl(self,idx):
        return self.get_cls(idx,'lensed')
    
    def get_delensed_cl(self,idx):
        return self.get_cls(idx,'delensed')
    
    def get_template_cl(self,idx):
        return self.get_cls(idx,'template')
    
    def run_job(self,imin,imax):
        jobs = np.arange(imin,imax+1)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Computing lensed sim {i} in {mpi.rank} processor")
            clss = self.get_lensed_cl(i)
            del clss
            print(f"Computing delensed sim {i} in {mpi.rank} processor")
            clss = self.get_delensed_cl(i)
            del clss            
        mpi.barrier()
        
    def get_spectra(self, which, imin,imax):
                
        spectra = []
        for i in tqdm(range(imin, imax+1), desc='Mean and STD of Bandpower', unit='simulation'):
            spectra.append(self.get_cls(i,which))
        
        return np.array(spectra)
    
    def get_stat(self,clbb,which,version,imin,imax,lmax=1000):
        spectra = self.get_spectra(which,imin,imax)
        if version == '1':
            return np.mean(spectra,axis=0),np.std(spectra,axis=0)
        elif version == '2':
            ells = self.ell
            ilmax = np.sum(ells <= lmax)
            newells = ells[...,:ilmax]
            mask = self.mask
            fsky = np.mean(mask**2)**2/np.mean(mask**4)
            ref_dof = np.array([np.sum(2 * self.b.get_ell_list(i) + 1) * fsky
                                    for i in range(self.b.get_n_bands())])[..., :ilmax]
            ref = self.b.bin_cell(clbb[:self.b.lmax+1])[...,:ilmax]
            spectra = np.array([self.get_cls(i,which)[3] for i in range(13)])[...,:ilmax]
            mean = spectra.mean(0)
            bias = mean -ref
            cov = (spectra - ref)[..., None] * (spectra-ref)[..., None, :]
            cov = cov.mean(0)
            std = np.sqrt(np.diagonal(cov,axis1=-1, axis2=-2))
            cl2dl_new = newells * (newells+1)/(2*np.pi)
            return newells, cl2dl_new, ref, mean, std

        
        
        
    def plot_bb(self,clbb,which,version, imin=0, imax=12 ,xmin=2, xmax=1000, ymin=None, ymax=10, save=False,filename='bb.png'):
        pl.figure(figsize=(8,8))
        
        
        if version=='1':
            ell = np.arange(len(clbb))
            cl2dl = ell * (ell + 1) / (2 * np.pi)
            mean , std = self.get_stat(clbb,which,version,imin,imax)
            pl.plot(ell,clbb*cl2dl,label='Fiducial')
            pl.errorbar(self.ell,mean*self.cl2dl,yerr=std*self.cl2dl, fmt='r.', markersize='10',label=f"{which}".capitalize())
        elif version=='2':
            ell,cl2dl,fiducial,mean,std = self.get_stat(clbb,which,version,imin,imax,)
            pl.plot(ell,fiducial*cl2dl,label='Fiducial')
            pl.errorbar(ell,mean*cl2dl,yerr=std*cl2dl,fmt='r.',markersize='10',label=f"{which}".capitalize())
        
        
        
        pl.xscale('log')
        pl.yscale('log')
        pl.xlabel('$\ell$', fontsize='20')
        pl.ylabel('$D_\ell^{BB}$', fontsize='20')
        pl.xticks(fontsize='20')
        pl.yticks(fontsize='20')
        pl.legend(fontsize='20')
        pl.grid()
        pl.xlim(xmin,xmax)
        pl.ylim(ymin,ymax)
        if save:
            pl.savefig(filename,bbox_inches='tight')

