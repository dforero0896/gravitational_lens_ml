#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import tifffile
import sys
from astropy.visualization import AsymmetricPercentileInterval, LogStretch, MinMaxInterval
import aplpy 
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp


def get_file_id(filename, delimiters = '_|\.|-'):
    id_ = [int(s) for s in re.split(delimiters, filename) if s.isdigit()][0]
    return id_
def build_generator_dataframe(id_label_df, directory):
    files = os.listdir(directory)
    ids = [
        get_file_id(filename)
        for filename in files
    ]
    df = pd.DataFrame()
    df['filenames'] = files
    df['labels'] = id_label_df.loc[ids, 'is_lens'].values.astype(int)
    df['ID'] = ids
    return df