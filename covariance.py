import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm
import pymaster

from plancklens.helpers import mpi

class SampleCov:
    
    def __init__(self,lib_dir,eff_lib,nside,binsize,lmin,lmax,include_bias,GS,bias_file):
        self.lib_dir = lib_dir
        self.eff_lib = eff_lib
        self.b = pymaster.NmtBin.from_nside_linear(nside,binsize)
        self.ell = self.b.get_effective_ells()
        
        self.lensed,self.l_s,self.delensed,self.d_s = self.eff_lib.get_stat
        self.bias = self.eff_lib.bias
        self.nell = len(self.ell)
        self.nsamp = self.eff_lib.n_sim
        self.include_bias = include_bias
        self.GS = GS
        self.bias_file = bias_file
        
        self.select = np.where((self.ell >= lmin)& (self.ell <= lmax))[0]
        self.imin = self.select[0]
        self.imax = self.select[-1]+1
        self.ncov = len(self.select)
    
        
        if self.include_bias:
            print("The Variance of bias is also added to the covariance")
        else:
            print("The Variance of bias is not considered")
        
        if mpi.rank == 0:
             os.makedirs(self.lib_dir,exist_ok=True)
                
        if self.GS:
            print("Saving bias variance")
            self.bais_cov = self.save_bias
        else:
            if os.path.isfile(self.bias_file):
                self.bias_cov = self.save_bias
            else:
                print("The covariance of bias is not found. Please run again with 'do_GS=1'")
                raise FileNotFoundError
            
        
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
        if self.include_bias:
            fname = os.path.join(self.lib_dir,'Cov_delen_bias_fid.pkl')
        else:
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
            
            
            if self.include_bias:
                cov_delensed += self.bias_cov
            
            pk.dump(cov_delensed,open(fname,'wb'))
            return cov_delensed
    
    @property
    def save_bias(self):
        if os.path.isfile(self.bias_file):
            return pk.load(open(self.bias_file,'rb'))
        else:
            cov_delensed = np.zeros([self.ncov,self.ncov])
            for i in tqdm(range(self.nsamp),desc='Covariance of delensed CMB spectra',unit='simulation'):
                delensed = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax]
                cov_delensed += delensed[None,:]*delensed[:,None]
            cov_delensed/=self.nsamp
            cov_delensed -= self.delensed_fid[None,:]*self.delensed_fid[:,None]
            
            pk.dump(cov_delensed,open(self.bias_file,'wb'))
            return cov_delensed
            
    
    def check_diag(self):
        return np.allclose(self.l_s,np.sqrt(np.diag(self.lensed_fid_cov)))
    
    def plot_spectra(self):
        plt.figure(figsize=(8,8))
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
        corr_mat