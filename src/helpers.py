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


def get_file_id(filename, delimiters = '_|\.|-'):
    id_ = [int(s) for s in re.split(delimiters, filename) if s.isdigit()][0]
    return id_
def build_generator_dataframe_old(id_label_df, directory):
    files = os.listdir(directory)
    ids = [
        get_file_id(filename)
        for filename in files
    ]
    df = pd.DataFrame()
    df['filenames'] = [os.path.realpath(os.path.join(directory, f)) for f in files]
    df['labels'] = id_label_df.loc[ids, 'is_lens'].values.astype(int)
    df['ID'] = ids
    return df

def build_generator_dataframe(id_label_df, directory):
    files = os.listdir(directory)
    files_new = []
    
    ids = id_label_df["ID"][:]
    extension = os.path.splitext(files[0])[1]
    for id_ in ids:
        pathfile = directory + "/image_" + str(id_) + "_multiband" + extension
        files_new.append(pathfile)
        
    df = pd.DataFrame()
    df['filenames'] = files_new
    df['labels'] = id_label_df.loc[:, 'is_lens'].values.astype(int)
    df['ID'] = ids
    return df