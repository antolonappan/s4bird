import os
import numpy as np
import healpy as hp
from plancklens.helpers import mpi
from plancklens import utils
import pickle as pk
from noise import NoiseMap_LB_white,  NoiseMap_s4_LAT
import camb
import pysm3 as pysm
from shutil import copyfile
import pysm3.units as u
from helper import clhash,hash_check
import lenspyx
from lenspyx.utils import camb_clfile,camb_clfile2

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
    def __init__(self,nside,filebase,filebase2=None,weights=None,beam_de=None,beam_con=None,prefix="exp_sims_"):
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
            
            lb_fl = hp.gauss_beam(np.radians(self.beam_de[0]/60),lmax=6143)
            s4_fl = hp.gauss_beam(np.radians(self.beam_de[1]/60),lmax=6143)
        
            w_exp = pk.load(open(self.weights,'rb'))
            self.w_lb_t = w_exp['LB']['T'] * utils.cli(lb_fl)
            self.w_lb_p = w_exp['LB']['P'] * utils.cli(lb_fl)
            self.w_s4_t = w_exp['S4']['T'] * utils.cli(s4_fl)
            self.w_s4_p = w_exp['S4']['P'] * utils.cli(s4_fl)
            
            self.fl = hp.gauss_beam(np.radians(self.beam_con/60),lmax=6143)
            
            
    def get_combined_field(self,idx,hdu):
        alm1 = hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx:04d}.fits"), hdu=hdu)
        alm2 = hp.read_alm(os.path.join(self.filebase2,f"{self.prefix}{idx:04d}.fits"), hdu=hdu)
        if hdu ==1:
            alm = hp.almxfl(hp.almxfl(alm1,self.w_lb_t) + hp.almxfl(alm2,self.w_s4_t), self.fl)
        else:
            alm = hp.almxfl(hp.almxfl(alm1,self.w_lb_p) + hp.almxfl(alm2,self.w_s4_p), self.fl)
        del (alm1,alm2)
        nanarray  = np.where(np.isnan(alm) == True)[0]
        alm[nanarray] = 0
        return alm
        
    
    def get_sim_tlm(self,idx):
        if not self.combination:
            return  hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx:04d}.fits"), hdu=1)
        else:
            return self.get_combined_field(idx,1)

    
    def get_sim_elm(self,idx):
        if not self.combination:
            return hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx:04d}.fits"), hdu=2)
        else:
            return self.get_combined_field(idx,2)
    
    def get_sim_blm(self,idx):
        if not self.combination:
            return hp.read_alm(os.path.join(self.filebase,f"{self.prefix}{idx:04d}.fits"), hdu=3)
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
            print("Noise Model: 1/f")
            self.noise_model =  NoiseMap_s4_LAT(noise_folder,self.nside,self.n_sim)
        else:
            print("Noise Model: white")
            self.noise_model = NoiseMap_LB_white(noise_folder,self.nside,self.nlev_t,self.nlev_p,self.n_sim)
            
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            
        print(f"using {self.infolder}, saved to {self.outfolder}")
        
    def make_map(self, idx):
        fname = os.path.join(self.outfolder,f"exp_sims_{idx:04d}.fits")
        
        if os.path.isfile(fname):
            print(f"{fname} already exist")
        else:
            maps = hp.read_map(os.path.join(self.infolder,f"cmb_sims_{idx:04d}.fits"),(0,1,2))
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

    def run_job(self,missing=None):
        jobs = missing if missing is not None else np.arange(self.n_sim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Making map-{i} in processor-{mpi.rank}")
            self.make_map(i)
            
class CMBLensed_old:
    
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
        
        
    
class CMBLensed:
    """
    Lensing class:
    It saves seeds, Phi Map and Lensed CMB maps
    
    """
    def __init__(self,outfolder,nsim,cl_path,scal_file,pot_file,len_file,sim_set,verbose=False):
        self.outfolder = outfolder
        self.cl_unl = camb_clfile2(os.path.join(cl_path, scal_file))
        self.cl_pot = camb_clfile2(os.path.join(cl_path, pot_file))
        self.cl_len = camb_clfile2(os.path.join(cl_path, len_file))
        self.nside = 2048
        self.lmax = 4096
        self.dlmax = 1024
        self.facres = 0
        self.verbose = verbose
        self.nsim = nsim
        self.sim_set = sim_set
        
        if sim_set == 1:
            mass_set = 1
        elif (sim_set == 2) or (sim_set == 3):
            mass_set = 2
        elif sim_set == 4:
            assert len_file is not None
            mass_set = 1
        else:
            raise ValueError
        
        self.mass_set = mass_set
        
        #folder for CMB
        self.cmb_dir = os.path.join(self.outfolder,f"CMB_SET{self.sim_set}")
        #folder for mass
        self.mass_dir = os.path.join(self.outfolder,f"MASS_SET{self.mass_set}") 
        
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
            os.makedirs(self.mass_dir,exist_ok=True) 
            os.makedirs(self.cmb_dir,exist_ok=True)
        
        
        fname = os.path.join(self.outfolder,'seeds.pkl')
        if (not os.path.isfile(fname)) and (mpi.rank == 0):
            seeds = self.get_seeds
            pk.dump(seeds, open(fname,'wb'), protocol=2)
        mpi.barrier()
        self.seeds = pk.load(open(fname,'rb'))
        
        
        # Here I saves a dictonary with the artibutes of this class and given Cls. 
        # So everytime when this instance run it checks for the same setup
        # If any artribute has changed from the previous run
        fnhash = os.path.join(self.outfolder, "lensing_sim_hash.pk")
        if (mpi.rank == 0) and (not os.path.isfile(fnhash)):
            pk.dump(self.hashdict(), open(fnhash, 'wb'), protocol=2)
        mpi.barrier()
        
        hash_check(pk.load(open(fnhash, 'rb')), self.hashdict())

    def hashdict(self):
        return {'nside':self.nside,
                'lmax':self.lmax,
                'cl_ee': clhash(self.cl_unl['ee']),
                'cl_pp': clhash(self.cl_pot['pp']),
                'cl_tt': clhash(self.cl_len['tt']),
               }
    @property
    def get_seeds(self):
        """
        non-repeating seeds
        """
        seeds =[]
        no = 0
        while no <= self.nsim-1:
            r = np.random.randint(11111,99999)
            if r not in seeds:
                seeds.append(r)
                no+=1
        return seeds
    
    def vprint(self,string):
        if self.verbose:
            print(string)
                  
    def get_phi(self,idx):
        """
        set a seed
        generate phi_LM
        Save the phi
        """
        fname = os.path.join(self.mass_dir,f"phi_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"Phi field from cache: {idx}")
            return hp.read_alm(fname)
        else:
            rNo = self.mass_set - 1
            np.random.seed(self.seeds[idx]-rNo)
            plm = hp.synalm(self.cl_pot['pp'], lmax=self.lmax + self.dlmax, new=True)
            hp.write_alm(fname,plm)
            self.vprint(f"Phi field cached: {idx}")
            return plm
        
    def get_kappa(self,idx):
        """
        generate deflection field
        sqrt(L(L+1)) * \phi_{LM}
        """
        der = np.sqrt(np.arange(self.lmax + 1, dtype=float) * np.arange(1, self.lmax + 2))
        return hp.almxfl(self.get_phi(idx), der)
    
    def get_unlensed_alm(self,idx):
        self.vprint(f"Synalm-ing the Unlensed CMB temp: {idx}")
        Cls = [self.cl_unl['tt'],self.cl_unl['ee'],self.cl_unl['tt']*0,self.cl_unl['te']]
        np.random.seed(self.seeds[idx]+self.sim_set)
        alms = hp.synalm(Cls,lmax=self.lmax + self.dlmax,new=True)
        return alms
    
    def get_gauss_lensed(self,idx):
        fname = os.path.join(self.cmb_dir,f"cmb_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"CMB Gaussian fields from cache: {idx}")
            return hp.read_map(fname,(0,1,2),dtype=np.float64)
        else:
            Cls = [self.cl_len['tt'],self.cl_len['ee'],self.cl_len['bb'],self.cl_len['te']]
            np.random.seed(self.seeds[idx])
            maps = hp.synfast(Cls,self.nside,self.lmax,pol=True)
            hp.write_map(fname,maps,dtype=np.float64)
            self.vprint(f"CMB Gaussian fields cached: {idx}")
            return maps
            
            

    
    def get_lensed(self,idx):
        fname = os.path.join(self.cmb_dir,f"cmb_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            self.vprint(f"CMB fields from cache: {idx}")
            return hp.read_map(fname,(0,1,2),dtype=np.float64)
        else:
            dlm = self.get_kappa(idx)
            Red, Imd = hp.alm2map_spin([dlm, np.zeros_like(dlm)], self.nside, 1, hp.Alm.getlmax(dlm.size))
            del dlm
            tlm,elm,blm = self.get_unlensed_alm(idx)
            del blm
            T  = lenspyx.alm2lenmap(tlm, [Red, Imd], self.nside, 
                                    facres=self.facres, 
                                    verbose=False)
            del tlm
            Q, U  = lenspyx.alm2lenmap_spin([elm, None],[Red, Imd], 
                                            self.nside, 2, facres=self.facres,
                                            verbose=False)
            del (Red, Imd, elm)
            hp.write_map(fname,[T,Q,U],dtype=np.float64)
            self.vprint(f"CMB field cached: {idx}")         
            return [T,Q,U]
        
        
    def run_job(self):
        jobs = np.arange(self.nsim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Lensing map-{i} in processor-{mpi.rank}")
            if self.sim_set == 4:
                NULL = self.get_gauss_lensed(i)
            else:
                NULL = self.get_lensed(i)
            del NULL
