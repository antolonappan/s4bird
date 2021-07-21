import os
import numpy as np
import healpy as hp
from plancklens.helpers import mpi
from plancklens import utils
import pickle as pk
from config import *
from noise import NoiseMap_LB_white,  NoiseMap_s4_LAT
import camb
import pysm3 as pysm
from shutil import copyfile
import pysm3.units as u


class s4bird_simbase(object):

    def __init__(self,nside):
        self.nside = nside

    def hashdict(self):
        return {'sim_lib': 'ffp10 lensed scalar cmb inputs, freq 0'}

    @staticmethod
    def get_sim_tlm(idx):
        pass

    @staticmethod
    def get_sim_elm(idx):
        pass

    @staticmethod
    def get_sim_blm(idx):
        pass
    
    def get_sim_tmap(self,idx):
        tmap = self.get_sim_tlm(idx)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap

    def get_sim_pmap(self,idx):
        elm = self.get_sim_elm(idx)
        blm = self.get_sim_blm(idx)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return Q ,U

class s4bird_sims_general(s4bird_simbase):
    def __init__(self,nside,filebase,filebase2=None,weights=None,beam_de=None,beam_con=None,prefix="cmbonly_"):
        super().__init__(nside)
        self.filebase = filebase
        self.filebase2 = filebase2
        self.weights = weights
        self.beam_de = beam_de
        self.beam_con = beam_con
        self.prefix = prefix
        
        self.combination = False
        self.w_lb = None
        self.w_s4 = None
        self.fl = None
        
        if self.filebase2 is None:
            print(f"Simulation uses {self.filebase}")
        else:
            assert self.weights is not None # weights for combaining two Experiments
            assert len(self.beam_de) == 2 # beam to deconvolve maps
            assert self.beam_con is not None # beam to convolve with the combined map
            print(f"Simulation uses a combination of {self.filebase} and {self.filebase2}")
            print(f"Simulation Warning: Default first filebase and first beam assumes LiteBird")
            self.combination = True

        
            w_exp = pk.load(open(self.weights,'rb'))
            w_lb_ = w_exp['lb']
            w_s4_ = w_exp['s4']
            w_total = w_lb_ + w_s4_
            
            lb_fl = hp.gauss_beam(np.radians(self.beam_de[0]/60),lmax=6143)
            s4_fl = hp.gauss_beam(np.radians(self.beam_de[1]/60),lmax=6143)
            self.w_lb = w_lb_/w_total * utils.cli(lb_fl)
            self.w_s4 = w_s4_/w_total * utils.cli(s4_fl)
            self.fl = hp.gauss_beam(np.radians(self.beam_con/60),lmax=6143)
            
            
    def get_combined_field(self,idx,hdu):
        alm1 = hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx}.fits"), hdu=hdu)
        alm2 = hp.read_alm(os.path.join(self.filebase2,f"{self.prefix}{idx}.fits"), hdu=hdu)
        alm = hp.almxfl(hp.almxfl(alm1,self.w_lb) + hp.almxfl(alm2,self.w_s4), self.fl)
        del (alm1,alm2)
        nanarray  = np.where(np.isnan(alm) == True)[0]
        alm[nanarray] = 0
        return alm
        
    
    def get_sim_tlm(self,idx):
        if not self.combination:
            return hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx}.fits"), hdu=1)
        else:
            return self.get_combined_field(idx,1)

    
    def get_sim_elm(self,idx):
        if not self.combination:
            return hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx}.fits"), hdu=2)
        else:
            return self.get_combined_field(idx,2)
    
    def get_sim_blm(self,idx):
        if not self.combination:
            return hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx}.fits"), hdu=3)
        else:
            return self.get_combined_field(idx,3)
    

    
class GaussSim:
    
    __slots__ = ["Cls","outfolder","nside","n_sim","seeds"]
    
    def __init__(self,cl_folder,cl_base,outfolder,nside,n_sim,seed_file=None):
        cl_file = os.path.join(cl_folder,f"{cl_base}_lensedCls.dat")
        print(f"Using {cl_file}")
        cl_len = utils.camb_clfile(cl_file)
        
        
        
        self.Cls = [cl_len['tt'],cl_len['ee'],cl_len['bb'],cl_len['te']*0]
        self.outfolder = outfolder
        self.nside = nside
        self.n_sim = n_sim
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            
        fname = os.path.join(self.outfolder,'seeds.pkl')
        seeds = [np.random.randint(11111,99999) for i in range(self.n_sim)]
        
        if (not os.path.isfile(fname)) and (mpi.rank == 0) and (seed_file == None):
            pk.dump(seeds, open(fname,'wb'), protocol=2)
        elif (os.path.isfile(fname)) and (mpi.rank==0):
            if len(pk.load(open(fname,'rb'))) != self.n_sim:
                print('The No of simulation is different from the No of seeds: Rewriting Seeds')
                pk.dump(seeds, open(fname,'wb'), protocol=2)
        else:
            pass
            
    
        mpi.barrier()
        
        
        if seed_file != None:
            print(f"Simulations use a seed file:{seed_file}")
            fname = seed_file
            
        self.seeds = pk.load(open(fname,'rb'))

    def make_map(self,idx):
        fname = os.path.join(self.outfolder,f"cmbonly_{idx}.fits")
        if os.path.isfile(fname):
            print(f"{fname} already exist")
        else:
            np.random.seed(self.seeds[idx])
            maps = hp.synfast(self.Cls,nside=self.nside,new=True)
            hp.write_map(fname,maps)
            del maps
        
    def run_job(self):
        jobs = np.arange(self.n_sim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Making map-{i} in processor-{mpi.rank}")
            self.make_map(i)        

            
class SimExperiment:
    
    __slots__ = ["infolder","outfolder","nside","mask","fwhm","nlev_t","nlev_p","n_sim","red_noise","noise_model"]
    
    def __init__(self,infolder,outfolder,nside,maskpath,fwhm,nlev_t,nlev_p,n_sim,noise_folder,red_noise=False,):
        self.infolder = infolder
        self.outfolder = outfolder
        self.nside = nside
        self.mask = hp.read_map(maskpath,verbose=False)
        self.fwhm = np.radians(fwhm/60.) 
        self.nlev_t = nlev_t
        self.nlev_p = nlev_p
        self.n_sim = n_sim
        self.red_noise = red_noise
        self.noise_model = None
        
        if red_noise:
            self.noise_model =  NoiseMap_s4_LAT(noise_folder,self.nside,self.n_sim)
        else:
            self.noise_model = NoiseMap_LB_white(noise_folder,self.nside,self.nlev_t,self.nlev_p,self.n_sim)
            
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            
        print(f"using {self.infolder}, saved to {self.outfolder}")
        
    def make_map(self, idx):
        fname = os.path.join(self.outfolder,f"cmbonly_{idx}.fits")
        
        if os.path.isfile(fname):
            print(f"{fname} already exist")
        else:
            maps = hp.read_map(os.path.join(self.infolder,f"cmbonly_{idx}.fits"),(0,1,2))
            if self.red_noise:
                print("simulation using 1/f noise")
                noise = self.noise_model.get_maps(idx)
            else:
                print('simulation using white noise')
                noise = self.noise_model.get_maps(idx)
            sm_maps = hp.smoothing(maps,self.fwhm) + noise
            del(maps,noise)
            alms = hp.map2alm([sm_maps[0]*self.mask,sm_maps[1]*self.mask,sm_maps[2]*self.mask])
            hp.write_alm(fname,alms)
            del alms

    def run_job(self):
        jobs = np.arange(self.n_sim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Making map-{i} in processor-{mpi.rank}")
            self.make_map(i)
            
class CMBLensed:
    
    def __init__(self,outfolder,nside,cl_folder,base,n_sim,seed_file=None,prefix='cmbonly_'):
        
        self.outfolder = outfolder
        self.cl_folder = cl_folder
        self.base = base
        self.nside = nside
        self.prefix = prefix
        self.n_sim = n_sim
        self.seed_file = seed_file
        
        
        if mpi.rank == 0:
            os.makedirs(self.cl_folder,exist_ok=True)
            os.makedirs(self.outfolder,exist_ok=True)
            

            
            if len(os.listdir(self.cl_folder))  < 2:
                print('Cl folder is Empty, Trying to execute CAMB')
                src = '/global/u2/l/lonappan/workspace/S4bird/ini/CAMB.ini'
                dst = f"{self.cl_folder}/CAMB.ini"
                print('    Coping Template')
                copyfile(src,dst)
                print('    Copied sucessfully')
                print('    Setting Output folder')
                with open(dst) as f:
                    lines = f.readlines()
                    
                lines[0] = f"output_root = {self.cl_folder}/{self.base}\n"
                with open(dst, "w") as f:
                    f.writelines(lines)
                
                print('    Executing CAMB')
                camb.run_ini(dst)
                print('    Cls Generated')
            else:
                print("Found Cls, Trying to use that")
                
        seeds = [np.random.randint(11111,99999) for i in range(self.n_sim)]

        fname = os.path.join(self.outfolder,'seeds.pkl')
        if (not os.path.isfile(fname)) and (mpi.rank == 0) and (seed_file == None):
            pk.dump(seeds, open(fname,'wb'), protocol=2)
        elif (os.path.isfile(fname)) and (mpi.rank==0):
            if len(pk.load(open(fname,'rb'))) != self.n_sim:
                print('The No of simulation is different from the No of seeds: Rewriting Seeds')
                pk.dump(seeds, open(fname,'wb'), protocol=2)
        else:
            pass
        
        mpi.barrier()
        
        
        if seed_file != None:
            print(f"Simulations use a seed file:{seed_file}")
            fname = seed_file
        self.seeds = pk.load(open(fname,'rb'))
    
    def sky(self,idx):
        cl_fname = os.path.join(self.cl_folder,f"{self.base}_lenspotentialCls.dat")
        cfg = {'c1':
                   {"class":"CMBLensed",
                    "cmb_spectra":f"{cl_fname}",
                    "cmb_seed" : self.seeds[idx]
                    }
                }
        return pysm.Sky(self.nside,component_config=cfg)
    
    def make_map(self,idx):
        fname = os.path.join(self.outfolder,f"{self.prefix}{idx}.fits")
        if os.path.isfile(fname):
            print(f"{fname} already exist")
        else:
            sky = self.sky(idx)
            maps = sky.get_emission(150 * u.GHz)
            maps_cmb = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(150*u.GHz))
            del maps
            hp.write_map(fname, maps_cmb)
            del maps_cmb
        
    
    def run_job(self):
        jobs = np.arange(self.n_sim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Making map-{i} in processor-{mpi.rank}")
            self.make_map(i) 
        
        
    

