import os
import toml
import pickle as pk
import numpy as np
from copy import deepcopy
from delens import DelensAndCl
from covariance import MeanAndCovariance
from likelihood import Delens_Theory
from likelihood import LH_simple3 as LH_simple


class Stat:
    
    def __init__(self,ini,lmin,lmax,do_ini=True):

        ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini/'
        inif = os.path.join(ini_dir,ini)
        self.lmin = lmin
        self.lmax = lmax

        ini_dic = toml.load(inif)

        dir_base = os.path.join(ini_dic['File']['base_folder'],ini_dic['File']['base_name'],'Stats')
        self.lib_dir = os.path.join(dir_base,f"S_{self.lmin}{self.lmax}_{ini_dic['Delens']['template']}")
        
        
        if do_ini:
            DC1_,DC2_,DC3_ = self.__ini_gen__(ini_dic)
            self.DC1 = DelensAndCl(DC1_)
            self.DC2 = DelensAndCl(DC2_)
            self.DC3 = DelensAndCl(DC3_)

            camb_ini = '/global/cscratch1/sd/lonappan/S4BIRD/CAMB/BBSims_params.ini'
            self.theory = Delens_Theory(camb_ini,self.DC2.pseudocl_lib.b.lmax,self.DC1.delens_lib.get_N0(0),30,2.16)

            self.mc = MeanAndCovariance(self.lib_dir,self.DC1,self.DC2,self.DC3,self.theory,self.lmin,self.lmax)

    
    def __ini_gen__(self,dic):
        sets = [[1,1],[2,3],[3,3]]
        dicts = []
        for set_i in sets:
            of = set_i[0]
            by = set_i[1]
            dict_i = deepcopy(dic)
            dict_i['OF']['set'] = of
            dict_i['BY']['set'] = by
            
            dicts.append(dict_i)
        return dicts
    
    def get_lensed_chain(self,nsamples=5000):
        fname = os.path.join(self.lib_dir,f'LensedChain_{nsamples}.pkl')
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            samples = LH_simple(self.mc,self.theory,self.lmin,self.lmax,'lensed').posterior(self.mc.get_biased('lensed').mean(axis=0))
            pk.dump(samples,open(fname,'wb'))
            return samples
    
    def get_delensed_chain(self,nsamples=5000):
        fname = os.path.join(self.lib_dir,f'DelensedChain_{nsamples}.pkl')
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            samples = LH_simple(self.mc,self.theory,self.lmin,self.lmax,'delensed').posterior(self.mc.debiased_wo_mc.mean(axis=0))
            pk.dump(samples,open(fname,'wb'))
            return samples
        
        
def get_efficency(lensed,delensed):
    return np.round_(100*(1 - np.mean(delensed/lensed)),2)