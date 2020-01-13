#!/usr/bin/env python
import numpy as np
import sys
data = np.loadtxt(sys.argv[1])
data[:,0] = data[:,0].astype(int)

np.savetxt(sys.argv[1], data, fmt='%i %.5f')
