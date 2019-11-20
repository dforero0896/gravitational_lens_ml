import os
import sys
import re
import numpy as np
import pandas as pd
import configparser
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from data_generator_function import TiffImageDataGenerator
import resnet_func as myf


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
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    #config.sections()
    
    print("\nConfiguration file:\n")
    for section in config.sections():
        print("Section: %s" % section)
        for options in config.options(section):
            print("  %s = %s" % (options,
                                    config.get(section, options)))


    if not bool(config['general'].getboolean('use_gpu')):
        sys.stdout.write('\nNot using GPU.\n')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    ###### Paths
    WORKDIR = config['general']['workdir']    
    #WORKDIR = os.path.abspath(sys.argv[2])
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    #SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = config['general']['train_multiband']
    #TRAIN_MULTIBAND = os.path.join(DATA, 'train_multiband')
    #TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

    image_catalog = pd.read_csv(os.path.join(DATA, 'datapack2.0train/image_catalog2.0train.csv'), comment='#', index_col=0)
    print('The shape of the image catalog: ' + str(image_catalog.shape) + "\n")  



    # Training parameters
    batch_size = config['trainparams'].getint('batch_size')  # orig paper trained all networks with batch_size=128
    epochs = config['trainparams'].getint('epochs')
    num_classes = 2
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
    version = config['trainparams'].getint('restnetversion')
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    ###### Create the dataframe containing filenames and labels.    
    # This is ok if we use weighted losses. #TODO: Weighted loss
    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col=0)
    dataframe_for_generator = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
    # Extract data proportions for loss weighting
    n_lens_clean = len(lens_df[lens_df['is_lens'] == True])
    n_nolens_clean = len(lens_df[lens_df['is_lens'] == False])
    equal_class_coeff = np.array([n_lens_clean/n_nolens_clean,1])
    natural_class_coeff = np.array([1000 * n_lens_clean/n_nolens_clean,1])
    
    ###### Split the TRAIN_MULTIBAND set into train and validation sets. Set test_size below!
    train_df, val_df = train_test_split(local_test_df, test_size=config['trainparams'].getfloat('test_fraction'), random_state=42)
    total_train = len(train_df)
    total_val = len(val_df)
    
    ###### Create Tiff Image Data Generator objects for train and validation
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          zca_epsilon=1e-06,
                                          rotation_range=10,
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

    ###### Create generators for Images and Labels
    train_data_gen = image_data_gen_train.image_generator_dataframe(train_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=batch_size, validation=False)
    
    val_data_gen = image_data_gen_val.image_generator_dataframe(train_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=batch_size, validation=True)
 
    ###### Obtain the shape of the input data (train images)
    temp_data_gen = image_data_gen_train.image_generator_dataframe(train_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=batch_size, validation=False)

    image, _ = next(temp_data_gen)
    input_shape = image[0].shape

    # Define correct bias to initialize
    output_bias = tf.keras.initializers.Constant(np.log(n_lens_clean/n_nolens_clean))
    ###### Create model
    if version == 2:
        model = myf.resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
    else:
        model = myf.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
    # Define metrics for the model.
    #metrics = [keras.metrics.TruePositives(name='tp'),
    #  keras.metrics.FalsePositives(name='fp'),
    #  keras.metrics.TrueNegatives(name='tn'),
    #  keras.metrics.FalseNegatives(name='fn'), 
    #  keras.metrics.BinaryAccuracy(name='accuracy'),
    #  keras.metrics.AUC(name='auc')]

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=myf.lr_schedule(0)),
                metrics=['accuracy'])
    model.summary()

    # Prepare model model saving directory.
    save_dir = os.path.join(RESULTS, 'checkpoints/resnet/')
    model_name = 'gravlens_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)
                                #save_weights_only=True

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(myf.lr_schedule)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # Define class weights for unevenly distributed (biased) dataset.
    if data_bias == 'natural':
        sys.stdout.write('Using natural data bias: 1000x more non lenses than lenses.\n')
        class_coeff = natural_class_coeff
    elif data_bias == 'none':
        sys.stdout.write('Using no data bias (simulate equal proportion among classes).\n')
        class_coeff = equal_class_coeff
    elif data_bias == 'raw':
        sys.stdout.write('Using the raw bias (no weights applied).\n')
        class_coeff = [1.,1.]
    else:
        raise NotImplementedError('data_bias must be either natural or none.')
    class_weights = {0:class_coeff[0], 1:class_coeff[1]}
    sys.stdout.write('Using weights: %s\n'%class_weights)

    ###### Train the ResNet
    print('Train the ResNet using real-time data augmentation.')
        
    history = model.fit_generator(train_data_gen,
                                steps_per_epoch=total_train,
                                epochs=epochs,
                                validation_data=val_data_gen,
                                validation_steps=total_val,
                                callbacks=callbacks,
                                class_weight= class_weights)
          
    # Score trained model.
    scores = model.evaluate_generator(val_data_gen, verbose=1, steps=total_val)
    model.save(os.path.join(RESULTS,model_name))
    with open(os.path.join(RESULTS,model_name.replace('h5', 'history')), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    main()
