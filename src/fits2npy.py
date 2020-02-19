#!/usr/bin/env python
from build_dataset import fits_to_npy
import sys
if __name__ == '__main__':
    ifile = sys.argv[1]
    odir = sys.argv[2]
    fits_to_npy(ifile, odir)
