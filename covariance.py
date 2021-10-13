import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm 
import seaborn as sns


class Cov:
    
    def __init__(self,case):
        dire = '/global/u2/l/lonappan/workspace/S4bird/Data/Cov'
        self.GS = pk.load(open(os.path.join(dire,f"GS_{case}.pkl"),'rb'))
        self.RS = pk.load(open(os.path.join(dire,f"RS_{case}.pkl"),'rb'))
        
        self.nsamp = len(self.GS['lensed'])
        self.nell = len(self.GS['lensed'][0])
        self.bias = self.calculate_bias()

        self.Rlensed,self.Rdelensed = self.calculate_spectra_mean(self.RS)
        self.Glensed,self.Gdelensed = self.calculate_spectra_mean(self.GS)
        
        self.Rcov_lensed,self.Rcov_delensed = self.calculate_spectra_cov(self.RS,'R')
        self.Gcov_lensed,self.Gcov_delensed = self.calculate_spectra_cov(self.GS,'G')
        
        self.delensed_debiased = self.Rdelensed-self.bias
        
        self.bias_cov = self.calculate_bias_cov()
        self.cov_delensed_debiased = self.calculate_real_cov()
        
        
    def calculate_bias(self):
        bias = np.zeros(self.nell)
        for i in tqdm(range(self.nsamp),desc='Calculating Bias',unit='simulation'):
            bias += (self.GS['delensed'][i] -self.GS['lensed'][i])    
        bias/=self.nsamp
        return bias
    
    def calculate_spectra_mean(self,SIM):
        lensed,delensed = np.zeros(self.nell),np.zeros(self.nell)
        for i in tqdm(range(self.nsamp),desc='Calculating Mean CMB spectra',unit='simulation'):
            lensed += SIM['lensed'][i]
            delensed += SIM['delensed'][i]
        lensed/=self.nsamp
        delensed/=self.nsamp
        return lensed,delensed
    
    def calculate_spectra_cov(self,SIM,which):
        cov_lensed,cov_delensed = np.zeros([self.nell,self.nell]),np.zeros([self.nell,self.nell])
        for i in tqdm(range(self.nsamp),desc='Calculating Covariance CMB spectra',unit='simulation'):
            lensed = SIM['lensed'][i]
            delensed = SIM['delensed'][i]
            cov_lensed += lensed[None,:]*lensed[:,None]
            cov_delensed += delensed[None,:]*delensed[:,None]
        cov_lensed/=self.nsamp
        cov_delensed/=self.nsamp
        
        cov_lensed -= self.Rlensed[None,:]*self.Rlensed[:,None] if which == 'R' else self.Glensed[None,:]*self.Glensed[:,None] 
        cov_delensed -= self.Rdelensed[None,:]*self.Rdelensed[:,None] if which == 'R' else self.Gdelensed[None,:]*self.Gdelensed[:,None]
    
        return cov_lensed,cov_delensed
    
    def get_corr(self,cov):
        corr = np.zeros(self.Rcov_lensed.shape)
        for i in range(self.Rcov_lensed.shape[0]):
            for j in range(self.Rcov_lensed.shape[1]):
                corr[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
        return corr
    
    def plot_corr(self,cov):
        corr = self.get_corr(cov)
        sns.heatmap(corr)
        
    def plot_spectra(self):
        plt.loglog(self.Rlensed,label='lensed')
        plt.loglog(self.Rdelensed,label='delensed')
        plt.loglog(self.delensed_debiased,label='delensed-debiased')
        plt.legend()
        
    
    def calculate_bias_cov(self,):
        cov = np.zeros([self.nell,self.nell]) 
        for i in tqdm(range(self.nsamp),desc='Calculating Covariance of Bias',unit='simulation'):
            lensed = self.GS['lensed'][i]
            delensed = self.GS['delensed'][i]
            cov += lensed[None,:]*delensed[:,None]
        
        cov /= self.nsamp
        cov -= self.Glensed[None,:]*self.Gdelensed[None,:]
        
        x = np.zeros(self.Gcov_lensed.shape)
        y = np.zeros(self.Gcov_lensed.shape)
        np.fill_diagonal(x,np.diag(self.Gcov_lensed))
        np.fill_diagonal(y,np.diag(self.Gcov_delensed))
        return x +  y #- (2*self.Glensed*self.Gdelensed*cov)
    
    def calculate_real_cov_old(self):
        cov = np.zeros([self.nell,self.nell]) 
        for i in tqdm(range(self.nsamp),desc='Calculating Real Covariance of CMB spectra',unit='simulation'):
            delensed = self.RS['delensed'][i]
            cov += delensed[None,:]*self.bias[:,None]
        
        cov /= self.nsamp
        cov -= self.Rdelensed[None,:]*self.bias[None,:]
        
        x = np.zeros(self.Gcov_lensed.shape)
        y = np.zeros(self.Gcov_lensed.shape)
        np.fill_diagonal(x,np.diag(self.Rcov_delensed))
        np.fill_diagonal(y,np.diag(self.bias_cov))
        return x + y #- 2*cov
    
    def calculate_real_cov(self):
        cov = np.zeros([self.nell,self.nell]) 
        for i in tqdm(range(self.nsamp),desc='Calculating Real Covariance of CMB spectra',unit='simulation'):
            delensed = self.RS['delensed'][i] - self.bias
            cov += delensed[None,:]*delensed[:,None]
        
        cov /= self.nsamp
        cov -= self.delensed_debiased[None,:]*self.delensed_debiased[None,:]
        
        x = np.zeros(self.Gcov_lensed.shape)

        np.fill_diagonal(x,np.diag(cov))
        return x #+ y #- 2*cov