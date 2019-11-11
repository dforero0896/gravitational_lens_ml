#!/usr/bin/env python
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsymmetricPercentileInterval, LogStretch, MinMaxInterval
from reproject import reproject_interp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import tifffile

def get_image_filename_from_id(id_, band, set_):
    fname = os.path.join(set_, '{0}/image{0}-{1}.fits'.format(band, id_))
    return fname
def build_image(id_, set_, bands = ['EUC_VIS', 'EUC_H', 'EUC_J', 'EUC_Y'], img_size=200, scale = 100):
    tables = []
    data = np.empty((img_size, img_size, len(bands)))
    for i, band in enumerate(bands):
        fname = get_image_filename_from_id(id_, band, set_)
        try:
            tables.append(fits.open(fname))
        except FileNotFoundError as fe:
            raise
        if band != 'EUC_VIS':
            band_data, data_footprint = reproject_interp(tables[i][0], tables[0][0].header)
        else:
            band_data = tables[0][0].data
        band_data[np.isnan(band_data)] = 0.
        interval = AsymmetricPercentileInterval(0.25, 99.75, n_samples=10000)
        vmin, vmax = interval.get_limits(band_data)
        stretch = MinMaxInterval() +  LogStretch()
        data[:,:,i] = stretch(((np.clip(band_data, -vmin*0.7, vmax))/(vmax)))
    for t in tables:
        t.close()
    return data.astype(np.float32)
def save_img_dataset(id_list, set_, outpath='.'):
    sys.stdout.write('Saving into directory %s\n'%os.path.realpath(outpath))
    for id_ in id_list:
        sys.stdout.write('Processing ID: %i\r'%id_)
        outname = os.path.join(outpath, 'image_%s_multiband.tiff'%id_)
        if os.path.isfile(outname):
            continue
        try:
            image = build_image(id_, set_)
            tifffile.imsave(outname, image)
        except FileNotFoundError as fe:
            sys.stdout.write(str(fe)+'\nContinuing...\n')
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('ERROR:\tPlease provide the path of the project directory.\nUSAGE:\t%s PROJECT_DIR\n'%sys.argv[0])
    WORKDIR=os.path.abspath(sys.argv[1])
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR,'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN = os.path.join(DATA, 'datapack2.0train/Public')
    TEST = os.path.join(DATA, 'datapack2.0test/Public')
    image_catalog = pd.read_csv(os.path.join(DATA, 'catalog/image_catalog2.0train.csv'), comment='#', index_col=0)
    image_catalog['is_lens'] = (image_catalog['mag_lens'] > 1.2) & (image_catalog['n_sources'] != 0)
    image_catalog[['ID', 'is_lens']].to_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index=False)
    train_outpath = os.path.join(DATA, 'train_multiband')
    test_outpath = os.path.join(DATA, 'test_multiband')
    if not os.path.isdir(test_outpath):
        os.mkdir(test_outpath)
    if not os.path.isdir(train_outpath):
        os.mkdir(train_outpath)
    save_img_dataset(image_catalog['ID'].values, TRAIN, train_outpath)
    save_img_dataset(image_catalog['ID'].values, TEST, test_outpath)
