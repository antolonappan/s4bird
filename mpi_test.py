from plancklens.helpers import mpi
import numpy as np

jobs = np.arange(4)
for i in jobs[mpi.rank::mpi.size]:
    print(f"Doing {i} in {mpi.rank}")