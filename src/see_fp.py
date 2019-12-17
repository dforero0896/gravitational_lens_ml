#!/usr/bin/env python
import tensorflow as tf
import data_generator_function
import helpers
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import configparser
import sys
matplotlib.rcParams["image.cmap"]="Greys"

if len(sys.argv) == 2:
    config_file = 'config_lesta_df.ini'
    model_name = sys.argv[1]
elif len(sys.argv) == 3:
    config_file = sys.argv[1]
    model_name = sys.argv[2]
else:
    sys.exit(
        'ERROR:\tUnexpected number of arguments.\nUSAGE:\t%s [CONFIG_FILE] MODEL_FILENAME'
        % sys.argv[0])
if not os.path.isfile(config_file):
    sys.exit('ERROR:\tThe config file %s was not found.' % config_file)
if not os.path.isfile(model_name):
    sys.exit('ERROR:\tThe model file %s was not found.' % model_name)
# Import configuration file
config = configparser.ConfigParser()
config.read(config_file)
WORKDIR = config['general']['workdir']
SRC = os.path.join(WORKDIR, 'src')
DATA = os.path.join(WORKDIR, 'data')
RESULTS = os.path.join(WORKDIR, 'results')
TRAIN = os.path.join(DATA, 'datapack2.0train/Public')
TEST = os.path.join(DATA, 'datapack2.0test/Public')
TRAIN_MULTIBAND = config['general']['train_multiband']
TRAIN_MULTIBAND_NOCLIP = os.path.join(DATA, 'train_multiband_noclip_bin')
TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')
CHECKPOINTS = os.path.join(RESULTS, 'checkpoints')
REPORT = os.path.join(WORKDIR, 'report')
print(os.path.isfile(model_name))
model = tf.keras.models.load_model(model_name)
lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col=0)
dataframe_for_generator = helpers.build_generator_dataframe(
    lens_df, TRAIN_MULTIBAND)
print(TRAIN_MULTIBAND)
total_examples = len(dataframe_for_generator)
# Extract bands from filename
bands = []
if 'VIS0' in model_name:
    bands.append(False)
elif 'VIS1' in model_name:
    bands.append(True)
if 'NIR000' in model_name:
    [bands.append(False) for i in range(3)]
elif 'NIR111' in model_name:
    [bands.append(True) for i in range(3)]
else:
    print("Couldn\'t extract used bands from filename, using all.")
    bands = [True, True, True, True]
bands = list(np.array(bands).reshape(-1))

n_val = 2000
train_df, val_df = train_test_split(dataframe_for_generator,
                                    test_size=n_val,
                                    random_state=42)
total_train = len(train_df)
total_val = len(val_df)
print(total_val)
image_data_gen_val = data_generator_function.TiffImageDataGenerator(
    dtype='float32')
roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(
    dataframe_for_generator,
    directory=TRAIN_MULTIBAND,
    x_col='filenames',
    y_col='labels',
    batch_size=n_val,
    validation=True,
    ratio=0.5,
    bands=bands,
    binary=True,
    get_ids=True,
    id_col='filenames')
images_val, labels_true, ids = next(roc_val_data_gen)
labels_score = model.predict(images_val,
                             steps=1,
                             verbose=2,
                             workers=16,
                             use_multiprocessing=True)
truth = (labels_true == 1)
predictions = (labels_score > 0.5).reshape(-1)

FP = []
for i in range(len(truth)):
    if not(truth[i]) and predictions[i]:
        FP.append(True)
    else:
        FP.append(False)
FP = np.array(FP)
fpr = np.mean(FP)

print(fpr)
fp_ids = ids[FP]
n_fp = len(fp_ids)
print(n_fp)
max_arr_side = 5
array_side = min(int(np.sqrt(n_fp)), max_arr_side)
fig, ax = plt.subplots(array_side, array_side, figsize=(10, 10))
for i, a in enumerate(ax.ravel()):
    img = np.load(fp_ids[i])
    a.imshow(img[:, :, 0])
    a.set_xticklabels([])
    a.set_yticklabels([])

plt.subplots_adjust(wspace=0, hspace=0)
#fig.tight_layout()
fig.savefig(os.path.join(WORKDIR, os.path.basename(
    model_name).replace('.h5', '_FALSE_POSITIVES.png')), dpi=100, bbox_inches='tight')
