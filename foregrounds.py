import pysm3
import pysm3.units as u
import healpy as hp
import numpy as np
from database import surveys

import os
import toml
from plancklens.helpers import mpi
from plancklens import utils
import argparse

ini_dir = '/global/u2/l/lonappan/workspace/S4bird/ini'


parser = argparse.ArgumentParser(description='ini')
parser.add_argument('inifile', type=str, nargs=1)
parser.add_argument('-dust', dest='dust',action='store_true',help='Make Dust maps')
parser.add_argument('-synch', dest='synch',action='store_true',help='Make Synchtron maps')
args = parser.parse_args()
ini = args.inifile[0]


ini_file = os.path.join(ini_dir,ini)
config = toml.load(ini_file)



map_config = config['Map']
file_config = config['File']
fg_config = config['FG']

lb_df = surveys().get_table_dataframe('LITEBIRD')

freq = list(lb_df.frequency)
beam = list(lb_df.fwhm)


nside = map_config['nside']
base = file_config['base_folder']
fg_dir = fg_config['folder']

strings = []

if args.dust:
    fg_path = os.path.join(base,fg_dir,'Dust')
    strings.append(fg_config['dust'])
if args.synch:
    fg_path = os.path.join(base,fg_dir,'Synchrotron')
    strings.append(fg_config['synchrotron'])
    
os.makedirs(fg_path,exist_ok=True)

sky = pysm3.Sky(nside=nside, preset_strings=strings)
print(f"Rendered Sky at {nside} NSIDE")

for v in freq:
    fname = os.path.join(fg_path,f"{strings[0]}_{int(v)}.fits")
    if not os.path.isfile(fname):
        print(f"Producing Maps at {v} GHz")
        map_v = sky.get_emission(v * u.GHz)
        map_v = map_v.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(v*u.GHz))
        hp.write_map(fname, map_v )
        del map_v

