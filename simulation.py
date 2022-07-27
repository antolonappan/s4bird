import os
import numpy as np
import healpy as hp
import mpi
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
from database import surveys
from fgbuster import harmonic_ilc_alm,CMB
from tqdm import tqdm

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
    

    
class SimExperimentFG:
    
    def __init__(self,infolder,outfolder,dnside,maskpath,fwhm,fg_dir,fg_str,table):
        
        self.infolder = infolder
        self.outfolder = outfolder
        self.lmax = (3*dnside)-1
        self.fwhm = np.radians(fwhm/60)
        self.fg_dir = fg_dir
        self.fg_str = fg_str
        
#         if mpi.rank == 0:
#             table = surveys().get_table_dataframe(table)
#             mask = hp.ud_grade(hp.read_map(maskpath,verbose=False),dnside)
#         else:
#             table = None
#             mask = None
#         mpi.barrier()
            
#         table = mpi.com.bcast(table, root=0)
#         mask = mpi.com.bcast(mask, root=0)
#         mpi.barrier()
#         self.mask = mask
#         self.table = table
        self.table = surveys().get_table_dataframe(table)
        self.mask = hp.ud_grade(hp.read_map(maskpath,verbose=False),dnside)
        self.dnside = dnside
        
        if mpi.rank == 0:
            os.makedirs(self.outfolder,exist_ok=True)
        print(f"using {self.infolder} and {self.fg_dir} saving to {self.outfolder}")
        
    def get_cmb(self,idx):
        fname = os.path.join(self.infolder,f"cmb_sims_{idx:04d}.fits")
        return hp.ud_grade(hp.read_map(fname,(0,1,2)),self.dnside)

    def get_fg(self,v):
        fname = os.path.join(self.fg_dir,f"{self.fg_str}_{int(v)}.fits")
        return hp.ud_grade(hp.read_map(fname,(0,1,2)),self.dnside)


    def get_noise(self,depth_i,depth_p):
        n_pix = hp.nside2npix(self.dnside)
        res = np.random.normal(size=(n_pix, 3))
        depth = np.stack((depth_i, depth_p, depth_p))
        depth *= u.arcmin * u.uK_CMB
        depth = depth.to(
            getattr(u, 'uK_CMB') * u.arcmin,
            equivalencies=u.cmb_equivalencies(0 * u.GHz))
        res *= depth.value / hp.nside2resol(self.dnside, True)
        return  res.T

    def get_total_alms(self,idx,v,n_t,n_p,beam):
        maps = hp.smoothing(self.get_cmb(idx)+self.get_fg(v),fwhm=np.radians(beam/60.)) + self.get_noise(n_t,n_p)
        alms = hp.map2alm(maps*self.mask)
        del maps
        beam = hp.gauss_beam(np.radians(beam/60),lmax=self.lmax,pol=True).T
        hp.almxfl(alms[0],1/beam[0],inplace=True)
        hp.almxfl(alms[1],1/beam[1],inplace=True)
        hp.almxfl(alms[2],1/beam[2],inplace=True)
        return alms
        

    def get_alms_arr(self,idx,v,n_t,n_p,beam):
        arr = []
        for i in tqdm(range(len(v)),desc="Making alms",unit='Freq'):
            arr.append(self.get_total_alms(idx,v[i],n_t[i],n_p[i],beam[i]))
        return np.array(arr)
    
    def get_comp_sep_alm(self,idx):
        fname = os.path.join(self.outfolder,f"exp_sims_{idx:04d}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname,(1,2,3))
        else:
            freqs = np.array(self.table.frequency)
            fwhm = np.array(self.table.fwhm)
            nlev_p = np.array(self.table.depth_p)
            nlev_t = nlev_p/np.sqrt(2)
            alms = self.get_alms_arr(idx,freqs,nlev_t,nlev_p,fwhm)
            instrument = INST(None,freqs)
            components = [CMB()]
            bins = np.arange(1000) * 10
            result = harmonic_ilc_alm(components, instrument,alms,bins)
            del alms
            alms = hp.smoothalm([result.s[0][0], result.s[0][1],result.s[0][2]],fwhm=self.fwhm)
            del result
            hp.write_alm(fname,alms)
            return alms
        
    def get_weights(self,idx):
        fname = os.path.join(self.outfolder,f"exp_weight_{idx:04d}.pkl")
        print(f"Getting Weights {idx}")
        if os.path.isfile(fname):
            return pk.load(open(fname,'rb'))
        else:
            freqs = np.array(self.table.frequency)
            fwhm = np.array(self.table.fwhm)
            nlev_p = np.array(self.table.depth_p)
            nlev_t = nlev_p/np.sqrt(2)
            alms = self.get_alms_arr(idx,freqs,nlev_t,nlev_p,fwhm)
            instrument = INST(None,freqs)
            components = [CMB()]
            bins = np.arange(1000) * 10
            result = harmonic_ilc_alm(components, instrument,alms,bins)
            w = result.W
            pk.dump(w,open(fname,'wb'))
            return w
        
    def run_job(self,nsim,weight=False):
        jobs = np.arange(nsim)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Component sep-{i} in processor-{mpi.rank}")
            if not weight:
                self.get_comp_sep_alm(i)
            else:
                self.get_weights(i)
    

        



class INST:
    
    def __init__(self,beam,frequency):
        self.Beam = beam
        self.fwhm = beam
        self.frequency = frequency
        
        
        
        
    
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
