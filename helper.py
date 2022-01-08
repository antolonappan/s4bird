import numpy as np
import healpy as hp
import os
import hashlib


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
            
            
def clhash(cl, dtype=np.float16):
    return hashlib.sha1(np.copy(cl.astype(dtype), order='C')).hexdigest()

def hash_check(hash1, hash2, ignore=['lib_dir', 'prefix'], keychain=[]):
    keys1 = hash1.keys()
    keys2 = hash2.keys()

    for key in ignore:
        if key in keys1: keys1.remove(key)
        if key in keys2: keys2.remove(key)

    for key in set(keys1).union(set(keys2)):
        v1 = hash1[key]
        v2 = hash2[key]

        def hashfail(msg=None):
            print("ERROR: HASHCHECK FAIL AT KEY = " + ':'.join(keychain + [key]))
            if msg is not None:
                print("   " + msg)
            print("   ", "V1 = ", v1)
            print("   ", "V2 = ", v2)
            assert 0

        if type(v1) != type(v2):
            hashfail('UNEQUAL TYPES')
        elif type(v2) == dict:
            hash_check( v1, v2, ignore=ignore, keychain=keychain + [key] )
        elif type(v1) == np.ndarray:
            if not np.allclose(v1, v2):
                hashfail('UNEQUAL ARRAY')
        else:
            if not( v1 == v2 ):
                hashfail('UNEQUAL VALUES')

def combine_mask(mask1,mask2):
    print(f"fsky of Mask1: {get_fsky(mask1)}")
    print(f"fsky of Mask2: {get_fsky(mask2)}")
    total = mask1+mask2
    total[np.where(total==1)] = 0
    total[np.where(total==2)] = 1
    print(f"fsky of Combined mask:{get_fsky(total)}")
    return total

def get_fsky(mask):
    return len(mask[mask==1])/len(mask)