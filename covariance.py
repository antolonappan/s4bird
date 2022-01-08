import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm
import pymaster

from plancklens.helpers import mpi


class SampleCOV:
    
    def __init__(self,lib_dir,folder,cl_dir,key,nsamp,nside,binsize,lmin,lmax):
        self.lib_dir = lib_dir
        self.folder = folder
        self.gauss_dir = os.path.join(folder,f"GS/{cl_dir}/C_l")
        self.real_dir = os.path.join(folder,f"RS/{cl_dir}/C_l")
        self.nsamp = nsamp
        self.lname = f"sims_lensed_"
        self.dname = f"sims_delensed_{key}_"
        self.b = pymaster.NmtBin.from_nside_linear(nside,binsize)
        self.ell = self.b.get_effective_ells()
        self.select = np.where((self.ell >= lmin)& (self.ell <= lmax))[0]
        self.ncov = len(self.select)
        self.imin = self.select[0]
        self.imax = self.select[-1]+1
        self.ells = self.ell[self.select]
        self.path = os.path.join(self.lib_dir,"Covariance")
        if mpi.rank == 0:
            os.makedirs(self.lib_dir,exist_ok=True)
            os.makedirs(self.path,exist_ok=True)
            

    
    def read_cl(self,fname):  
        cl =  pk.load(open(fname,'rb'))
        cl_BB = cl[3][self.imin:self.imax]
        return cl_BB
    
    
    def get_lensed_cl(self,idx,which='real'):
        if which == 'real':
            folder = self.real_dir
        elif which == 'gauss':
            folder = self.gauss_dir
        else:
            raise ValueError
            
        fname = os.path.join(folder,f"{self.lname}{idx}.pkl")    
        return self.read_cl(fname)
    
    def get_delensed_cl(self,idx,which='real'):
        if which == 'real':
            folder = self.real_dir
        elif which == 'gauss':
            folder = self.gauss_dir
        else:
            raise ValueError
        
        fname = os.path.join(folder,f"{self.dname}{idx}.pkl")  
        return self.read_cl(fname)

    
    def lensed_mean(self,which='real'):
        fname = os.path.join(self.path, f"lensed_mean_{which}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:                
            folder = self.real_dir if which == 'real' else self.gauss_dir
            cl_mean = np.zeros(self.ncov)
            for i in range(self.nsamp):
                cl_mean += self.get_lensed_cl(i,which)
            cl_mean/=self.nsamp
            pk.dump(cl_mean,open(fname,'wb'))
        return cl_mean
    
    @property
    def lensed_fid(self):
        return self.lensed_mean('real')
    

    def delensed_mean(self,which='real'):
        fname = os.path.join(self.path, f"delensed_mean_{which}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            folder = self.real_dir if which == 'real' else self.gauss_dir
            cl_mean = np.zeros(self.ncov)
            for i in range(self.nsamp):
                cl_mean += self.get_delensed_cl(i,which)
            cl_mean/=self.nsamp
            pk.dump(cl_mean,open(fname,'wb'))
        return cl_mean
    
    @property
    def bias(self):
        fname = os.path.join(self.path, f"bias.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            cl_mean = np.zeros(self.ncov)
            for i in range(self.nsamp):
                cl_mean += (self.get_delensed_cl(i,'gauss') - self.get_lensed_cl(i,'gauss'))
            cl_mean/=self.nsamp
            pk.dump(cl_mean,open(fname,'wb'))
        return cl_mean
    
    @property
    def delensed_fid(self):
        fname = os.path.join(self.path, f"delensed_fid.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            bias = self.bias
            cl_mean = np.zeros(self.ncov)
            for i in range(self.nsamp):
                cl_mean += (self.get_delensed_cl(i,'real') - bias)
            cl_mean/=self.nsamp
            pk.dump(cl_mean,open(fname,'wb'))
            
        return cl_mean
        
    
    @property
    def lensed_fid_cov(self):
        fname = os.path.join(self.path, f"lensed_fid_cov.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            mean = self.lensed_fid
            cov_lensed = np.zeros([self.ncov,self.ncov])
            for i in tqdm(range(self.nsamp),desc='Covariance of lensed CMB spectra',unit='simulation'):
                lensed = self.get_lensed_cl(i,'real')
                cov_lensed += lensed[None,:]*lensed[:,None]
            cov_lensed/=self.nsamp
            cov_lensed -= mean[None,:]*mean[:,None]
            pk.dump(cov_lensed,open(fname,'wb'))

        return cov_lensed
    
    @property
    def delensed_fid_cov(self):
        fname = os.path.join(self.path, f"delensed_fid_cov.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            mean = self.delensed_fid
            cov_delensed = np.zeros([self.ncov,self.ncov])
            for i in tqdm(range(self.nsamp),desc='Covariance of delensed CMB spectra',unit='simulation'):
                delensed = self.get_delensed_cl(i,'real')
                bias = self.get_delensed_cl(i,'gauss') - self.get_lensed_cl(i,'gauss')
                delensed_debias = delensed-bias
                cov_delensed += delensed_debias[None,:]*delensed_debias[:,None]

            cov_delensed/=self.nsamp
            cov_delensed -= mean[None,:]*mean[:,None]
            pk.dump(cov_delensed,open(fname,'wb'))
        return cov_delensed
    
    
    def plot_spectra(self):
        plt.figure(figsize=(10,12))
        plt.loglog(self.ells,self.lensed_fid,label='Lensed')
        plt.loglog(self.ells,self.delensed_fid,label='Delensed')
        plt.loglog(self.ells,self.delensed_mean('real'),label='Biased Delensed')
        plt.legend()
        
    def corr(self,mat):
        sha = mat.shape
        corr_mat = np.zeros(sha)
        for i in range(sha[0]):
            for j in range(sha[1]):
                corr_mat[i,j] = mat[i,j]/np.sqrt(mat[i,i]*mat[j,j])      
        return corr_mat
        
    
    def plot_stat(self,savefig=False):
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
        im = axs[1,1].imshow(self.corr(self.delensed_fid_cov))
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)      
        
        

class SampleCov:
    
    def __init__(self,lib_dir,eff_lib,nside,binsize,lmin,lmax,do_gs,bias_cov_f):
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
        self.bias_cov_f = bias_cov_f
        self.do_gs = do_gs
        
        if mpi.rank == 0:
             os.makedirs(self.lib_dir,exist_ok=True)
    
        if do_gs:
            NULL = self.delensed_fid_cov_old
            
        
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
        
    def change_test(self):
        cov_delensed = np.zeros([self.ncov,self.ncov])
        cov_delensed_b = np.zeros([self.ncov,self.ncov])
        delensed_mean = np.zeros(self.ncov)
        delensed_b_mean = np.zeros(self.ncov)
        for i in tqdm(range(self.nsamp),desc='Testing',unit='simulation'):
            delensed = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax]
            delensed_b = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax] - self.bias[self.imin:self.imax]
            delensed_mean += delensed
            delensed_b_mean += delensed_b
            cov_delensed += delensed[None,:]*delensed[:,None]
            cov_delensed_b += delensed_b[None,:]*delensed_b[:,None]
        delensed_mean/=self.nsamp
        cov_delensed/=self.nsamp
        cov_delensed -= delensed_mean[None,:]*delensed_mean[:,None]
        
        delensed_b_mean/=self.nsamp
        cov_delensed_b/=self.nsamp
        cov_delensed_b -= delensed_b_mean[None,:]*delensed_b_mean[:,None]
        
        #plt.loglog(delensed_mean,label="biased delens")
        #plt.loglog(delensed_b_mean,label="debiased delens")
        plt.loglog(np.diag(cov_delensed),label="biased")
        plt.scatter(np.arange(self.ncov), np.diag(cov_delensed_b),label="debiased",marker='*')
        plt.legend()
            
        
            
    @property
    def delensed_fid_cov_old(self):
        if self.do_gs:
            fname = self.bias_cov_f
        else:
            fname = os.path.join(self.lib_dir,f"Cov_delen_fid_{self.imin}_{self.imax}.pkl")
            
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            delensed_mean = np.zeros(self.ncov)
            lensed_mean = np.zeros(self.ncov)
            
            cov_delensed = np.zeros([self.ncov,self.ncov])
            cov_lensed = np.zeros([self.ncov,self.ncov])
            cov_del_len = np.zeros([self.ncov,self.ncov])
            
            if self.do_gs:
                for i in tqdm(range(self.nsamp),desc='Covariance of Bias',unit='simulation'):
                    delensed = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax]
                    lensed = self.eff_lib.lib_pcl.get_lensed_cl(i)[3][self.imin:self.imax]
                    
                    delensed_mean += delensed
                    cov_delensed += delensed[None,:]*delensed[:,None]
                    
                    lensed_mean += lensed
                    cov_lensed += lensed[None,:]*lensed[:,None]
                    
                    cov_del_len += lensed[None,:]*delensed[:,None]
                
                delensed_mean/=self.nsamp
                cov_delensed/=self.nsamp
                cov_delensed -= delensed_mean[None,:]*delensed_mean[:,None]
                
                lensed_mean/=self.nsamp
                cov_lensed/=self.nsamp
                cov_lensed -= lensed_mean[None,:]*lensed_mean[:,None]
                
                cov_del_len/=self.nsamp
                cov_del_len -= lensed_mean[None,:]*delensed_mean[:,None]
                
                cov_delen = np.zeros([self.ncov,self.ncov])
                cov_len = np.zeros([self.ncov,self.ncov])
                
                np.fill_diagonal(cov_delen,np.diag(cov_delensed))
                np.fill_diagonal(cov_len,np.diag(cov_lensed))
                
                res = cov_delen + cov_len - 2* cov_del_len
                
                cov_delensed = res
                
            else:
                for i in tqdm(range(self.nsamp),desc='Covariance of delensed CMB spectra',unit='simulation'):
                    delensed = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax] - self.bias[self.imin:self.imax]
                    cov_delensed += delensed[None,:]*delensed[:,None]
                cov_delensed/=self.nsamp
                cov_delensed -= self.delensed_fid[None,:]*self.delensed_fid[:,None]
            
            pk.dump(cov_delensed,open(fname,'wb'))
            return cov_delensed
        
    @property    
    def delensed_fid_cov(self):
        fname = os.path.join(self.lib_dir,f"Cov_delen_fid_{self.imin}_{self.imax}.pkl")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            cov_delensed = np.zeros([self.ncov,self.ncov])
            cov_delen_bias = np.zeros([self.ncov,self.ncov])

            for i in tqdm(range(self.nsamp),desc='Covariance of with corr',unit='simulation'):
                delensed = self.eff_lib.lib_pcl.get_delensed_cl(i)[3][self.imin:self.imax]
                cov_delensed += delensed[None,:]*delensed[:,None]
                cov_delen_bias += delensed[None,:]*self.bias[self.imin:self.imax][:,None]
            cov_delensed/=self.nsamp
            cov_delen_bias/=self.nsamp
            cov_delensed -= self.delensed[self.select][None,:]*self.delensed[self.select][:,None]
            cov_delen_bias -= self.delensed[self.select][None,:]*self.bias[self.imin:self.imax][:,None]

            Var_delens = np.zeros([self.ncov,self.ncov])
            Var_bias = np.zeros([self.ncov,self.ncov])
            np.fill_diagonal(Var_delens, np.diagonal(cov_delensed))
            np.fill_diagonal(Var_bias, np.diagonal(self.bias_cov))
            final = Var_delens + Var_bias - 2*cov_delen_bias
            pk.dump(final,open(fname,'wb'))
            return final
        
        
            
    
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
        im = axs[1,1].imshow(self.corr(self.delensed_fid_cov))
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
        if savefig:
            plt.savefig(fname,bbox_inches='tight')
            
    @property 
    def bias_cov(self):
        try:
            return pk.load(open(self.bias_cov_f,'rb'))
        except:
            raise FileNotFoundError
            
class MeanAndCovariance:
    
    def __init__(self,lib_dir,biased,unbiased,biased2,theory,lmin,lmax,nsim=1000):
        self.lib_dir = lib_dir
        os.makedirs(lib_dir,exist_ok=True)
        
        self.biased = biased
        self.unbiased = unbiased
        self.biased2 = biased2
        
        self.theory = theory
        self.ell = self.biased.pseudocl_lib.ell
        
        self.select = np.where((self.ell >= lmin)& (self.ell <= lmax))[0]
        self.nsim = nsim
        self.get_biased()
        self.get_unbiased()
        self.get_biased2()
        
        

    
    def get_biased(self,which=None):
        def creturn(dic):
            if which is None:
                pass
            else:
                return dic[which]
        
        fname = os.path.join(self.lib_dir,f'biased_{self.nsim}.pkl')
        if os.path.isfile(fname):
            return creturn(pk.load(open(fname,'rb')))
        else:
            lensed = []
            delensed = []
            for i in tqdm(range(self.nsim),desc='Biased spectra Array', unit='simulation'):
                lensed.append(self.biased.pseudocl_lib.get_lensed_cl(i))
                delensed.append(self.biased.pseudocl_lib.get_delensed_cl(i))
            dic = {'lensed':np.array(lensed),'delensed':np.array(delensed)}
            pk.dump(dic,open(fname,'wb'))
            return creturn(dic)
        
    def get_unbiased(self,which=None):
        def creturn(dic):
            if which is None:
                pass
            else:
                return dic[which]
        
        fname = os.path.join(self.lib_dir,f'unbiased_{self.nsim}.pkl')
        if os.path.isfile(fname):
            return creturn(pk.load(open(fname,'rb')))
        else:
            lensed = []
            delensed = []
            for i in tqdm(range(self.nsim),desc='Unbiased spectra Array', unit='simulation'):
                lensed.append(self.unbiased.pseudocl_lib.get_lensed_cl(i))
                delensed.append(self.unbiased.pseudocl_lib.get_delensed_cl(i))
            dic = {'lensed':np.array(lensed),'delensed':np.array(delensed)}
            pk.dump(dic,open(fname,'wb'))
            
            return creturn(dic)
        
    def get_biased2(self,which=None):
        def creturn(dic):
            if which is None:
                pass
            else:
                return dic[which]
        
        fname = os.path.join(self.lib_dir,f'biased2_{self.nsim}.pkl')
        if os.path.isfile(fname):
            return creturn(pk.load(open(fname,'rb')))
        else:
            lensed = []
            delensed = []
            for i in tqdm(range(self.nsim),desc='Biased2 spectra Array', unit='simulation'):
                lensed.append(self.biased2.pseudocl_lib.get_lensed_cl(i))
                delensed.append(self.biased2.pseudocl_lib.get_delensed_cl(i))
            dic = {'lensed':np.array(lensed),'delensed':np.array(delensed)}
            pk.dump(dic,open(fname,'wb'))
            return creturn(dic)
    
    @property
    def bias_array(self):
        biased_lensed = self.get_biased2('lensed')
        biased_delensed = self.get_biased2('delensed')
        unbiased_lensed = self.get_unbiased('lensed')
        unbiased_delensed = self.get_unbiased('delensed')
        
        df_biased = biased_delensed - biased_lensed
        df_unbiased = unbiased_delensed - unbiased_lensed
        
        return df_biased - df_unbiased
    
    @property
    def mcbias_array(self):
        lmax = self.unbiased.pseudocl_lib.b.lmax
        unbiased_lensed = self.get_unbiased('lensed')
        unbiased_delensed = self.get_unbiased('delensed')
        theory_delensed = (self.theory.delensed_bb * self.theory.fl**2) + self.theory.nlevp
        theory_lensed = (self.theory.lensed_bb * self.theory.fl**2) + self.theory.nlevp
        df_theory = self.unbiased.pseudocl_lib.b.bin_cell(theory_delensed-theory_lensed)
        df_unbiased = unbiased_delensed - unbiased_lensed
        
        return df_unbiased - df_theory
    
    def cov(self,arr):
        mean = arr.mean(axis=0)[self.select]
        ncov = len(mean)
        cov = np.zeros((ncov,ncov))
        for i in tqdm(range(self.nsim),desc='Covariance', unit='simulation'):
            arr_i = arr[i][self.select]
            cov += arr_i[None,:]*arr_i[:,None]
        cov/=self.nsim
        cov -= mean[None,:]*mean[:,None]
        return cov
    
    def cov_cross(self,arr1,arr2):
        mean1 = arr1.mean(axis=0)[self.select]
        mean2 = arr2.mean(axis=0)[self.select]
        ncov = len(mean1)
        cov = np.zeros((ncov,ncov))
        for i in tqdm(range(self.nsim),desc='Covariance', unit='simulation'):
            arr1_i = arr1[i][self.select]
            arr2_i = arr2[i][self.select]
            cov += arr1_i[None,:]*arr2_i[:,None]
        cov/=self.nsim
        cov -= mean1[None,:]*mean2[:,None]
        return cov 
    
    @property
    def lensed_cov_fid(self):
        return self.cov(self.get_biased('lensed'))
        
    @property
    def delensed_cov_fid(self):
        return self.cov(self.get_unbiased('delensed'))
    
    @property
    def debiased_w_mc(self):
        return self.get_biased('delensed') - self.bias_array - self.mcbias_array
    
    @property
    def debiased_wo_mc(self):
        return self.get_biased('delensed') - self.bias_array
    
    @property
    def delensed_cov_w_mc(self):
        return self.cov(self.debiased_w_mc)
    
    @property
    def delensed_cov_wo_mc(self):
        return self.cov(self.debiased_wo_mc)

    def corr(self,mat):
        sha = mat.shape
        corr_mat = np.zeros(sha)
        for i in range(sha[0]):
            for j in range(sha[1]):
                corr_mat[i,j] = mat[i,j]/np.sqrt(mat[i,i]*mat[j,j])      
        return corr_mat
    
    def plot_corr(self,mat):
        plt.imshow(self.corr(mat),cmap='GnBu')
        plt.colorbar()   