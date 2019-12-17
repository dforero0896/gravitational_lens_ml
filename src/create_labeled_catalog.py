#!/usr/bin/env python
import pandas as pd
import configparser
import os
import sys
import numpy as np
from helpers import get_file_id
if len(sys.argv) != 2:
    sys.exit(
        'ERROR: Unexpected number of arguments.\nUSAGE: %s CONFIG_FILE' % sys.argv[0])
config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)
WORKDIR = config['general']['workdir']
DATA = os.path.join(WORKDIR, 'data')
TRAIN = os.path.join(DATA, 'datapack2.0train/Public')
RESULTS = os.path.join(WORKDIR, 'results')
# Get ID of images in the raw train set.
file_id_train = np.array([get_file_id(f) for f in os.listdir(
    os.path.join(TRAIN, 'EUC_VIS'))], dtype=int)
# Load catalog
image_catalog = pd.read_csv(os.path.join(
    DATA, 'catalog/image_catalog2.0train.csv'), comment='#', index_col=0)
# Get IDs of missing images.
missing_img = np.setdiff1d(image_catalog.ID.values,
                           file_id_train, assume_unique=False)
# Add 'is_lens' flag to the catalog according to
# http://metcalf1.difa.unibo.it/DATA3/evaluation.pdf
image_catalog['is_lens'] = (image_catalog['mag_lens'] > 1.2) & (
    image_catalog['n_sources'] != 0)
# Add a flag to lines with no corresponding image
image_catalog['img_exists'] = True
image_catalog['img_exists'].loc[image_catalog['ID'].isin(missing_img)] = False
# Remove duplicate IDs
image_catalog = image_catalog.drop_duplicates('ID')
print('Number of lenses: %i' % image_catalog['is_lens'].sum())
# Save a new catalog with just ID, is_lens.
image_catalog[['ID',
               'is_lens']][image_catalog['img_exists']].to_csv(os.path.join(
                   RESULTS, 'lens_id_labels.csv'),
    index=False)