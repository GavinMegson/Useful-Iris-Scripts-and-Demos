import h5py
import numpy as np
import os

fs = []
names = []
for filename in os.listdir("./data_out"):
    names.append(filename)
    if filename.endswith(".hdf5"):
        f = h5py.File(filename, r)
        fs.append(f)
