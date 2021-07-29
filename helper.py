import numpy as np
import healpy as hp
from plancklens.helpers import mpi
import os


class maps2alm:
    
    def __init__(self,infolder,outfolder,base,nsims):
        self.infolder = infolder
        self.outfolder = outfolder
        self.base = base
        self.nsims = nsims
        os.makedirs(self.outfolder,exist_ok=True)
            
    def map2alm(self,idx):
        infname = os.path.join(self.infolder,f"{self.base}_{idx}.fits")
        outfname = os.path.join(self.outfolder,f"{self.base}_{idx}.fits")
        if not os.path.isfile(outfname):
            alms = hp.map2alm(hp.read_map(infname,field=(0,1,2)))
            hp.write_alm(outfname,alms)
            del alms
    
    def run_job(self):
        jobs = np.arange(self.nsims)
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Noise alms-{i} in processor-{mpi.rank}")
            self.map2alm(i)

if __name__ == '__main__':
    
    scr = os.environ['SCRATCH']
    inpath = os.path.join(scr,'S4BIRD','LiteBird_s4mask','Noise')
    outpath = os.path.join(scr,'S4BIRD','LiteBird_s4mask','NoiseAlm')

    m2a = maps2alm(inpath,outpath,'noiseonly',100)
    m2a.run_job()