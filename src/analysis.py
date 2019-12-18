import os   
import sys
import re
import configparser
import pickle
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from data_generator_function import TiffImageDataGenerator
tf.debugging.set_log_device_placement(False)

def get_file_id(filename, delimiters='_|\\.|-'):
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

def main():
    if len(sys.argv) != 2:
        config_file = 'config.ini'
    else:
        config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        sys.exit('ERROR:\tThe config file %s was not found.'%config_file)
    
    sys.stdout.write('\nNot using GPU.\n')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    

    if len(tf.config.experimental.list_physical_devices('GPU')):
        print('GPU found. Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
        print(tf.config.experimental.list_physical_devices('GPU'))
        if(len(tf.config.experimental.list_logical_devices('GPU'))):
            print('Logical GPU found. Num logical GPUs Available: ', len(tf.config.experimental.list_logical_devices('GPU')))
            print(tf.config.experimental.list_logical_devices('GPU'))
    else:
        print("No GPU found")

    if len(tf.config.experimental.list_physical_devices('CPU')):
        print('Physical CPU found. Num Physical CPUs Available: ', len(tf.config.experimental.list_physical_devices('CPU')))
        print(tf.config.experimental.list_physical_devices('CPU'))
        if(len(tf.config.experimental.list_logical_devices('CPU'))):
            print('Logical CPU found. Num logical CPUs Available: ', len(tf.config.experimental.list_logical_devices('CPU')))
            print(tf.config.experimental.list_logical_devices('CPU'))
    else:
        print("No CPU found")

    config = configparser.ConfigParser()
    config.read(config_file)
    
    print("\nConfiguration file:\n")
    for section in config.sections():
        print("Section: %s" % section)
        for options in config.options(section):
            print("  %s = %s" % (options, config.get(section, options)))
            
    ###### Paths
    WORKDIR = config['general']['workdir']    
    #WORKDIR = os.path.abspath(sys.argv[2])
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    #SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = config['general']['train_multiband']
    #TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

    image_catalog = pd.read_csv(os.path.join(DATA, 'catalog/image_catalog2.0train.csv'), comment='#', index_col=0)
    print('The shape of the image catalog: ' + str(image_catalog.shape) + "\n")  

    # Training parameters
    batch_size = config['trainparams'].getint('batch_size')  # orig paper trained all networks with batch_size=128
    epochs = config['trainparams'].getint('epochs')
    num_classes = 1
    data_bias = 'none'
    # Model parameter
    # ----------------------------------------------------------------------------
    #           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
    # Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
    #           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
    # ----------------------------------------------------------------------------
    # ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
    # ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
    # ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
    # ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
    # ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
    # ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
    # ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
    # --------------------------------------------------------------------------- 
    n = config['trainparams'].getint('n')
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = config['trainparams'].getint('resnetversion')
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # Model name, depth and version
    
    # This is ok if we use weighted losses.
    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col=0)
    dataframe_for_generator = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
   
    ###### Split the TRAIN_MULTIBAND set into train and validation sets. Set test_size below!
    train_df, val_df = train_test_split(dataframe_for_generator, test_size=config['trainparams'].getfloat('test_fraction'), random_state=42)
    total_train = len(train_df)
    total_val = len(val_df)
    print("The number of objects in the whole training sample is: ", total_train)
    print("The number of objects in the whole validation sample is: ", total_val)
    test_fraction = float(config["trainparams"]["test_fraction"])
    print("The test fraction is: ", test_fraction)
    if config['trainparams']['subsample_train'] == 'total':
        subsample_train = total_train
        subsample_val = total_val
    else:
        try:
            subsample_train = int(config['trainparams']['subsample_train'])
            subsample_val = int(subsample_train*test_fraction/(1.-test_fraction))
        except:
            raise ValueError('subsample_train should be \'total\' or int.')
    
    print("The number of objects in the training subsample is: ", subsample_train)
    print("The number of objects in the validation subsample is: ", subsample_val)
    train_steps_per_epoch = int(subsample_train//batch_size)
    val_steps_per_epoch = int(subsample_val//batch_size)
    print("The number of training steps is: ", train_steps_per_epoch)
    print("The number of validation steps is: ", val_steps_per_epoch)
    
    augment_train_data = bool(int(config['trainparams']['augment_train_data']))
    ###### Create Tiff Image Data Generator objects for train and validation
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          zca_epsilon=1e-06,
                                          rotation_range=0,
                                          width_shift_range=0.0,
                                          height_shift_range=0.0,
                                          brightness_range=(0.8, 1.1),
                                          shear_range=0.0,
                                          zoom_range=(0.9, 1.01),
                                          channel_shift_range=0.0,
                                          fill_mode='wrap',
                                          cval=0.0,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          rescale=None,
                                          preprocessing_function=None,
                                          data_format='channels_last',
                                          dtype='float32')

    image_data_gen_val = TiffImageDataGenerator(dtype='float32')
 
    bands = [config['bands'].getboolean('VIS0'), 
        config['bands'].getboolean('NIR1'),
        config['bands'].getboolean('NIR2'),
        config['bands'].getboolean('NIR3')]
    print("The bands are: ", bands)
    ###### Create generators for Images and Labels
    ratio = 0.5
    train_data_gen = image_data_gen_train.prop_image_generator_dataframe(train_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=batch_size, validation=not(augment_train_data), ratio=ratio,
                                bands=bands)
    
    val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=batch_size, validation=True, ratio=ratio,
                                bands=bands)
 
    roc_val_data_gen = image_data_gen_val.prop_image_generator_dataframe(val_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=subsample_val, validation=True, ratio=ratio,
                                bands=bands)
    
    ###### Obtain model from the saving directory
    model_type = 'RN%dv%d' % (depth, version)
    
    model_name = '%s_Tr%i_Te%i_bs%i_ep%.03d_aug%i_VIS%i_NIR%i%i%i_DB%s_ratio%.01f_dropout.h5' % (model_type,
                                                                        subsample_train,
                                                                        subsample_val,
                                                                        batch_size,
                                                                        epochs,
                                                                        int(augment_train_data),
                                                                        config['bands'].getint('VIS0'), 
                                                                        config['bands'].getint('NIR1'),
                                                                        config['bands'].getint('NIR2'),
                                                                        config['bands'].getint('NIR3'),
                                                                        config['trainparams']['data_bias'],
                                                                        ratio)
    model = tf.keras.models.load_model(os.path.join(RESULTS, model_name))
    model.summary()
    history_path = os.path.join(RESULTS, model_name.replace('h5', 'history'))
    ## History
    with open(history_path, 'rb') as file_pi:
        history = pickle.load(file_pi)

    ## Checkpoints
    save_dir = os.path.join(RESULTS, 'checkpoints/resnet/')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    ### Plots
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(range(len(history['loss'])),
            history['val_loss'],
            label='Validation loss',
            marker='o',
            c='b')
    ax1.plot(range(len(history['loss'])),
            history['loss'],
            label='Training loss',
            marker='o',
            c='r')
    ax1.set_ylim([0,1])
    ax2.plot(range(len(history['loss'])),
            history['val_acc'],
            label='Validation accuracy',
            marker='o',
            c='b',
            ls='--',
            fillstyle='none')
    ax2.plot(range(len(history['loss'])),
            history['acc'],
            label='Training accuracy',
            marker='o',
            c='r',
            ls='--',
            fillstyle='none')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc=(-0.1, 1))
    ax2.legend(loc=(0.9, 1))
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    plt.gcf()
    plt.savefig(os.path.join(RESULTS, 'plots/' + os.path.basename(history_path).replace('.history', '.png')),
                dpi=200)
    
    plt.figure(2)
    train_auc = np.array(history['auc'])
    val_auc = np.array(history['val_auc'])
    plt.plot(np.arange(0, len(val_auc)), train_auc, 'ob', label='Train')
    plt.plot(np.arange(0, len(val_auc)), val_auc, 'or', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(RESULTS, 'plots/AUC_' + os.path.basename(history_path).replace('.history', '.png')),
                dpi=200)
    print("history keys:\n", history.keys())
    ##Score
    #scores = model.evaluate_generator(val_data_gen, verbose=2, steps=val_steps_per_epoch)
    ##Roc curve 
    images_val, labels_true = next(roc_val_data_gen)
    labels_score = model.predict(images_val, batch_size=subsample_val, verbose=2)
    fpr, tpr, thresholds = roc_curve(np.ravel(labels_true), np.ravel(labels_score))
    print(fpr)
    print(tpr)

    plt.figure(3)
    plt.plot(fpr, tpr, 'or', label='Validation')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1])
    plt.legend()
    plt.savefig(os.path.join(RESULTS, 'plots/ROCsklearn_' + os.path.basename(history_path).replace('.history', '.png')),
                dpi=200)
    
    #print('Test loss:', scores[0])
    #print('Test accuracy:', scores[1])

if __name__ == '__main__':
    main()
