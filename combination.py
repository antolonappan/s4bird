import healpy as hp
import numpy as np
import os
import toml
from plancklens import utils
import mpi


class Experiment:
    def __init__(self,ini,red):

        
        mpi.barrier()
        ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini/'
        config = toml.load(os.path.join(ini_dir,ini))
        self.noise_map = os.path.join(config["File"]["base_folder"],config["File"]["base_name"],'Noise','noise_sims_')
        self.noise_alm = os.path.join(config["File"]["base_folder"],config["File"]["base_name"],'Noise','Alms','sims_alms_')
        self.beam = float(config["Map"]["beam"])
        self.gauss_beam = hp.gauss_beam(np.radians(self.beam/60),lmax=6143,pol=True).T
        fnoise = 1 + ((np.arange(len(self.gauss_beam[0]))/200)**-5)
        fnoise[0] = fnoise[1]
        self.fnoise = fnoise
        self.red = bool(red)
        
        self.libdir = os.path.join(config["File"]["base_folder"],config["File"]["base_name"])
        if mpi.rank ==0:
            os.makedirs(self.libdir,exist_ok=True)
        
    def get_noise_map(self,i):
        fname = f"{self.noise_map}{i:04d}.fits"
        return hp.read_map(fname,(0,1,2))
    
    def get_noise_alm(self,i):
        fname = f"{self.noise_alm}{i:04d}.fits"
        return hp.read_alm(fname,(1,2,3))
    
    def get_noise_alm_beam_deconv(self,i):
        alms = self.get_noise_alm(i)
        if not self.red:
            new_alms =  [ hp.almxfl(alms[0],utils.cli(self.gauss_beam[0])),
                          hp.almxfl(alms[1],utils.cli(self.gauss_beam[1])),
                          hp.almxfl(alms[2],utils.cli(self.gauss_beam[2]))
                         ]
        else:
             new_alms =  [ hp.almxfl(alms[0],utils.cli(self.gauss_beam[0])*self.fnoise),
                           hp.almxfl(alms[1],utils.cli(self.gauss_beam[1])*self.fnoise),
                           hp.almxfl(alms[2],utils.cli(self.gauss_beam[2])*self.fnoise)
                         ]                      
        del alms
        return new_alms
    
    def mpi_mean(self):
        fname = os.path.join(self.libdir,f"mean_noise_alm_{int(mpi.size)}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname,(1,2,3))
        else:
            if not mpi.size > 100:
                raise NotImplementedError
            mpi.barrier()
            data = self.get_noise_alm_beam_deconv(mpi.rank)
            tlm = np.abs(data[0])**2
            elm = np.abs(data[1])**2
            blm = np.abs(data[2])**2

            if mpi.rank == 0:
                total_tlm = np.zeros_like(tlm)
                total_elm = np.zeros_like(elm)
                total_blm = np.zeros_like(blm)
            else:
                total_tlm = None
                total_elm = None
                total_blm = None
            
            mpi.com.Reduce(tlm,total_tlm, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(elm,total_elm, op=mpi.mpi.SUM,root=0)
            mpi.com.Reduce(blm,total_blm, op=mpi.mpi.SUM,root=0)
            
            if mpi.rank == 0:
                mean = [total_tlm/mpi.size,total_elm/mpi.size,total_blm/mpi.size]
                hp.write_alm(fname,mean)
            
            mpi.barrier()
class W:
    def __init__(self,first,second):
        self.first = first
        self.second = second

    def weight(self,nlm):
        return np.nan_to_num(1/nlm)

    def get_w(self):
        w1 = self.weight(self.first)
        w2 = self.weight(self.second)
        total = w1+w2
        return w1/total , w2/total

class Combine:
    def __init__(self,ini):
        self.ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini/'
        config = toml.load(os.path.join(self.ini_dir,ini))
        file_conf = config["File"]
        map_conf = config["Map"]
        self.set = int(map_conf['set'])
        self.nsims = int(map_conf['nsims'])
        self.beam = np.radians(float(map_conf['beam'])/60)
        self.filebase = os.path.join(file_conf['base_folder'],
                                     file_conf['base_name'],
                                     f"SIM_SET{self.set}",
                                     "Maps")
        if mpi.rank == 0:
            os.makedirs(self.filebase,exist_ok=True)
        
        
        self.first = self.exp(map_conf["map1"])
        self.second = self.exp(map_conf["map2"])
        
        if self.first.nside != self.second.nside:
            raise ValueError
        self.nside = self.first.nside
        
        self.apply_mask_to = None
        self.mask = self.get_mask

        
    def exp(self,inifile):
        config = toml.load(os.path.join(self.ini_dir,inifile))
        map_conf = config['Map']
        file_conf = config['File']
        
        class Exp:
            beam = float(map_conf['beam'])
            maskpath = map_conf['mask']
            nside = map_conf['nside']
            filebase = os.path.join(file_conf['base_folder'],file_conf['base_name'])
            simulation = os.path.join(filebase,f"SIM_SET{self.set}","Maps")
            noise_mean = os.path.join(filebase,'mean_noise_alm_1000.fits')
                
        return Exp
    
    def get_fsky(self,mask):
        return len(mask[mask==1])/len(mask)
    
    
    @property
    def get_mask(self):
        mask1 = hp.read_map(self.first.maskpath)
        mask2 = hp.read_map(self.second.maskpath)
        fsky1 = self.get_fsky(mask1)
        fsky2 = self.get_fsky(mask2)
        print(f"Mask 1: fsky = {fsky1}")
        print(f"Mask 2: fsky = {fsky2}")
        if fsky1 < fsky2:
            mask = mask1
            self.apply_mask_to = 'second'
            del mask2
            print("Choosing Mask 1")
        else:
            mask = mask2
            self.apply_mask_to = 'first'
            del mask1
            print("Choosing Mask 2")
        
        return mask
    
    def deconv_map(self,path,beamsize):
        beam = hp.gauss_beam(np.radians(beamsize/60),lmax=6143,pol=True).T
        alms = hp.read_alm(path,(1,2,3))
        hp.almxfl(alms[0],utils.cli(beam[0]),inplace=True)
        hp.almxfl(alms[1],utils.cli(beam[1]),inplace=True)
        hp.almxfl(alms[2],utils.cli(beam[2]),inplace=True)
        del beam
        return alms
    def deconv_map_and_apply_mask(self,path,beamsize):
        beam = hp.gauss_beam(np.radians(beamsize/60),lmax=6143,pol=True).T
        maps = hp.alm2map(hp.read_alm(path,(1,2,3)),nside=self.nside) * self.mask
        alms = hp.map2alm(maps)
        del maps
        hp.almxfl(alms[0],utils.cli(beam[0]),inplace=True)
        hp.almxfl(alms[1],utils.cli(beam[1]),inplace=True)
        hp.almxfl(alms[2],utils.cli(beam[2]),inplace=True)
        del beam
        return alms
    
    def get_alm_to_combine(self,i):
        fname1 = os.path.join(self.first.simulation,f"exp_sims_{i:04d}.fits")
        fname2 = os.path.join(self.second.simulation,f"exp_sims_{i:04d}.fits")
        if self.apply_mask_to == "first":
            alms1 = self.deconv_map_and_apply_mask(fname1,self.first.beam)
            alms2 = self.deconv_map(fname2,self.second.beam)
        elif self.apply_mask_to == "second":
            alms1 = self.deconv_map(fname1,self.first.beam)
            alms2 = self.deconv_map_and_apply_mask(fname2,self.second.beam)
        else:
            raise NotImplementedError
        
        return alms1,alms2
    
    def get_combined_alm(self,i):
        fname = os.path.join(self.filebase,f"exp_sims_{i:04d}.fits")
        if os.path.isfile(fname):
            return hp.read_alm(fname,(1,2,3))
        else:
            mean_noise1 = hp.read_alm(self.first.noise_mean,(1,2,3))
            mean_noise2 = hp.read_alm(self.second.noise_mean,(1,2,3))

            w_tlm_1,w_tlm_2 = W(mean_noise1[0],mean_noise2[0]).get_w()
            w_elm_1,w_elm_2 = W(mean_noise1[1],mean_noise2[1]).get_w()
            w_blm_1,w_blm_2 = W(mean_noise1[2],mean_noise2[2]).get_w()

            del (mean_noise1,mean_noise2)

            alms1, alms2 = self.get_alm_to_combine(i)

            tlm = (alms1[0]*w_tlm_1) + (alms2[0]*w_tlm_2)
            elm = (alms1[1]*w_elm_1) + (alms2[1]*w_elm_2)
            blm = (alms1[2]*w_blm_1) + (alms2[2]*w_blm_2)
            hp.write_alm(fname,hp.smoothalm([tlm,elm,blm],fwhm=self.beam))
        
    def run_job(self):
        jobs = np.arange(self.nsims)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Combining maps {i} in Processor-{mpi.rank}")
            alms = self.get_combined_alm(i)
            del alms

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-red',dest='red',action='store_true')
    parser.add_argument('-job',dest='job',action='store_true')
    parser.add_argument('-comb',dest='comb',action='store_true')
    args = parser.parse_args()
    ini = args.inifile[0]
    
    
    if args.job:
        exp = Experiment(ini,args.red)
        exp.mpi_mean()
        
    if args.comb:
        exp = Combine(ini)
        exp.run_job()
    