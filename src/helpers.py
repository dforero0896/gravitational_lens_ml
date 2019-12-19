#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re
import sys



def get_file_id(filename, delimiters = '_|\.|-'):
    id_ = [int(s) for s in re.split(delimiters, filename) if s.isdigit()][0]
    return id_
def build_generator_dataframe(id_label_df, directory):
    files = os.listdir(directory)
    files_new = []
    existing_id = []
    ids = id_label_df.index
    extension = os.path.splitext(files[0])[1]
    for id_ in ids:
        pathfile = directory + "/image_" + str(id_) + "_multiband" + extension
        if not os.path.isfile(pathfile):
            continue
        files_new.append(pathfile)
        existing_id.append(id_)
        
    df = pd.DataFrame()
    df['filenames'] = files_new
    df['labels'] = id_label_df.loc[existing_id, 'is_lens'].values.astype(int)
    df['ID'] = existing_id
    return df
