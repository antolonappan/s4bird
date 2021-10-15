import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm
import pymaster

from plancklens.helpers import mpi

class SampleCov:
    
    def __init__(self,lib_dir,eff_lib,nside,binsize,lmin,lmax):
        self.lib_dir = lib_dir
        self.eff_lib = eff_lib
        self.b = pymaster.NmtBin.from_nside_linear(nside,binsize)
        self.ell = self.b.get_effective_ells()
        
        self.lensed,self.l_s,self.delensed,self.d_s = self.eff_lib.get_stat
        self.bias = self.eff_lib.bias
        self.nell = len(self.ell)
        self.nsamp = self.eff_lib.n_sim
        
        
        self.select = np.where((self.ell >= lmin)& (self.ell <= lmax))[0]
        self.imin = self.select[0]
        self.imax = self.select[-1]+1
        self.ncov = len(self.select)
    

        
        if mpi.rank == 0:
             os.makedirs(self.lib_dir,exist_ok=True)
            
        
    @property    
    def lensed_fid(self):
        fname = os.path.join(self.lib_dir,f"C_len_fid_{self.imin}_{self.imax}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            lensed = self.lensed[self.imin:self.imax]
            pk.dump(lensed,open(fname,'wb'))
            print(f"lensed len:{len(lensed)}")
            return lensed
    
    @property
    def delensed_fid(self):
        fname = os.path.join(self.lib_dir,f"C_delen_fid_{self.imin}_{self.imax}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            delensed = self.delensed[self.imin:self.imax]-self.bias[self.imin:self.imax]
            pk.dump(delensed,open(fname,'wb'))
            print(f"delensed len:{len(delensed)}")
            return delensed
    
    @property
    def lensed_fid_cov(self):
        fname = os.path.join(self.lib_dir,f"Cov_len_fid_{self.imin}_{self.imax}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            cov_lensed = np.zeros([self.ncov,self.ncov])#[self.nell,self.nell])
            for i in tqdm(range(self.nsamp),desc='Covariance of lensed CMB spectra',unit='simulation'):
                lensed = self.eff_lib.lib_pcl.get_lensed_cl(i)[3][self.imin:self.imax]
                cov_lensed += lensed[None,:]*lensed[:,None]
            cov_lensed/=self.nsamp
            cov_lensed -= self.lensed_fid[None,:]*self.lensed_fid[:,None]
            pk.dump(cov_lensed,open(fname,'wb'))
            return cov_lensed
            
    @property
    def delensed_fid_cov(self):
        fname = os.path.join(self.lib_dir,f"Cov_delen_fid_{self.imin}_{self.imax}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            cov_delensed = np.zeros([self.ncov,self.ncov])
            for i in tqdm(range(self.nsamp),desc='Covariance of delensed CMB spectra',unit='simulation'):
                delensed = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax] - self.bias[self.imin:self.imax]
                cov_delensed += delensed[None,:]*delensed[:,None]
            cov_delensed/=self.nsamp
            cov_delensed -= self.delensed_fid[None,:]*self.delensed_fid[:,None]
            
            pk.dump(cov_delensed,open(fname,'wb'))
            return cov_delensed
            
    
    def check_diag(self):
        return np.allclose(self.l_s,np.sqrt(np.diag(self.lensed_fid_cov)))
    
    def plot_spectra(self):
        plt.figure(figsize=(10,12))
        plt.loglog(self.eff_lib.lib_pcl.ell,self.lensed_fid,label='Lensed')
        plt.errorbar(self.eff_lib.lib_pcl.ell,self.delensed_fid,yerr=self.d_s,fmt='o',label='Delensed')
        plt.loglog(self.eff_lib.lib_pcl.ell,self.delensed,label='Biased Delensed')
        plt.legend()
        
    def corr(self,mat):
        sha = mat.shape
        corr_mat = np.zeros(sha)
        for i in range(sha[0]):
            for j in range(sha[1]):
                corr_mat[i,j] = mat[i,j]/np.sqrt(mat[i,i]*mat[j,j])      
        return corr_mat
        
    
    def plot_stat(self,savefig=False):
        fname = os.path.join(self.lib_dir,f"mat_check_{self.imin}_{self.imax}.png")
        def check_pos(cov):
            return f"Positive definite: {np.all(np.linalg.eigvals(cov) > 0)}"
        def check_det(cov):
            return f"Determinant: {np.linalg.det(cov):.2e}"
        
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0,0].plot(self.ell[self.select],np.linalg.eigvals(self.lensed_fid_cov))
        axs[0,0].plot(np.zeros(1),label=check_pos(self.lensed_fid_cov),c='k')
        axs[0,0].plot(np.zeros(1),label=check_det(self.lensed_fid_cov),c='k')
        axs[0,0].semilogy()
        axs[0,0].set_xlabel("N")
        axs[0,0].set_ylabel("E")
        axs[0,0].legend()
        axs[0,1].plot(self.ell[self.select],np.linalg.eigvals(self.delensed_fid_cov))
        axs[0,1].set_xlabel("N")
        axs[0,1].plot(np.zeros(1),label=check_pos(self.delensed_fid_cov),c='k')
        axs[0,1].plot(np.zeros(1),label=check_det(self.delensed_fid_cov),c='k')
        axs[0,1].semilogy()
        axs[0,1].legend()
        axs[1,0].imshow(self.corr(self.lensed_fid_cov))
        axs[1,1].imshow(self.corr(self.delensed_fid_cov))
        if savefig:
            plt.savefig(fname,bbox_inches='tight')
        