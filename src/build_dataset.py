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
def build_image(id_, set_, bands = ['EUC_VIS', 'EUC_H', 'EUC_J', 'EUC_Y'], img_size=200, scale = 100, clip = True):
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
        if clip:
            interval = AsymmetricPercentileInterval(0.25, 99.75, n_samples=100000)
            vmin, vmax = interval.get_limits(band_data)
            stretch = MinMaxInterval() +  LogStretch()
            data[:,:,i] = stretch(((np.clip(band_data, -vmin*0.7, vmax))/(vmax)))
        else:
            stretch =  LogStretch() + MinMaxInterval()
            data[:,:,i] = stretch(band_data)
    for t in tables:
        t.close()
    return data.astype(np.float32)
def save_img_dataset(id_list, set_, outpath='.', clip = False, overwrite = False):
    sys.stdout.write('Saving into directory %s\n'%os.path.realpath(outpath))
    sys.stdout.write('clip = %s\noverwrite = %s\n'%(clip, overwrite))
    for id_ in id_list:
        sys.stdout.write('Processing ID: %i\r'%id_)
        outname = os.path.join(outpath, 'image_%s_multiband.tiff'%id_)
        if os.path.isfile(outname) and not overwrite:
            continue
        try:
            image = build_image(id_, set_, clip = clip)
            #tifffile.imsave(outname, image)
            np.save(outname.replace('tiff', 'npy'), image) 
        except FileNotFoundError as fe:
            sys.stdout.write(str(fe)+'\nContinuing...\n')
        
if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit('ERROR:\tPlease provide the path of the project directory.\nUSAGE:\t%s PROJECT_DIR CLIP_PREPROCESS? OUT_DIR_NAME OVERWRITE? PARALLEL?\n'%sys.argv[0])
    clip = bool(int(sys.argv[2]))
    outdir = sys.argv[3]
    overwrite = bool(int(sys.argv[4]))
    parallel = bool(int(sys.argv[5]))
    if parallel:
        from mpi4py import MPI
        size = MPI.COMM_WORLD.Get_size()   # Size of communicator
        rank = MPI.COMM_WORLD.Get_rank()   # Ranks in communicator
        name = MPI.Get_processor_name()    # Node where this MPI process runs
    WORKDIR=os.path.abspath(sys.argv[1])
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR,'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN = os.path.join(DATA, 'datapack2.0train/Public')
    TEST = os.path.join(DATA, 'datapack2.0test/Public')
    image_catalog = pd.read_csv(os.path.join(DATA, 'catalog/image_catalog2.0train.csv'), comment='#', index_col=0)
    image_catalog['is_lens'] = (image_catalog['mag_lens'] > 1.2) & (image_catalog['n_sources'] != 0)
    #image_catalog[['ID', 'is_lens']].to_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index=False)
    train_outpath = os.path.join(DATA, 'train_%s'%outdir)
    test_outpath = os.path.join(DATA, 'test_%s'%outdir)
    if not os.path.isdir(test_outpath):
        os.mkdir(test_outpath)
    if not os.path.isdir(train_outpath):
        os.mkdir(train_outpath)
    id_list = image_catalog['ID'].values
    if parallel:
        new_id_list = np.array_split(id_list, size)
        id_list = new_id_list[rank]
        print('Process %i has %i files to go through'%(rank, len(id_list)))
    save_img_dataset(id_list, TRAIN, train_outpath, clip = clip, overwrite = overwrite)
    save_img_dataset(id_list, TEST, test_outpath, clip = clip, overwrite = overwrite)
    if parallel:
        MPI.Finalize()
