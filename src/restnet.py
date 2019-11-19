import os
import sys
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_generator_function import TiffImageDataGenerator
import restnet_func as myf


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
    ###### Paths
    WORKDIR = os.path.abspath(sys.argv[2])
    sys.stdout.write('Project directory: %s\n'%WORKDIR)
    #SRC = os.path.join(WORKDIR, 'src')
    DATA = os.path.join(WORKDIR, 'data')
    RESULTS = os.path.join(WORKDIR, 'results')
    TRAIN_MULTIBAND = os.path.join(DATA, 'train_multiband')
    #TEST_MULTIBAND = os.path.join(DATA, 'test_multiband')

    image_catalog = pd.read_csv(os.path.join(DATA, 'datapack2.0train/image_catalog2.0train.csv'), comment='#', index_col=0)
    print(image_catalog.shape)
    
    # Training parameters
    #batch_size = 32  # orig paper trained all networks with batch_size=128
    epochs = 1
    num_classes = 2

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
    n = 3
    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
        
    lens_df = pd.read_csv(os.path.join(RESULTS, 'lens_id_labels.csv'), index_col=0)
    local_test_df = build_generator_dataframe(lens_df, TRAIN_MULTIBAND)
    
    ###### Split the TRAIN_MULTIBAND set into train and validation sets. Set test_size below!
    train_df, val_df = train_test_split(local_test_df, test_size=0.1, random_state=42)
    total_train = len(train_df)
    total_val = len(val_df)
    
    ###### Create Tiff Image Data Generator objects for train and validation
    image_data_gen_train = TiffImageDataGenerator(featurewise_center=False,
                                          samplewise_center=False,
                                          featurewise_std_normalization=False,
                                          samplewise_std_normalization=False,
                                          zca_whitening=False,
                                          zca_epsilon=1e-06,
                                          rotation_range=90,
                                          width_shift_range=0.0,
                                          height_shift_range=0.0,
                                          brightness_range=(0.8, 1.1),
                                          shear_range=0.0,
                                          zoom_range=(0.9, 1.1),
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
                                y_col='labels', batch_size=1, validation=False)
    
    val_data_gen = image_data_gen_val.image_generator_dataframe(train_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=1, validation=True)
 
    ###### Obtain the shape of the input data (train images)
    temp_data_gen = image_data_gen_train.image_generator_dataframe(train_df,
                                directory=TRAIN_MULTIBAND,
                                x_col='filenames',
                                y_col='labels', batch_size=1, validation=False)

    image, _ = next(temp_data_gen)
    input_shape = image[0].shape

    ###### Create model
    if version == 2:
        model = myf.resnet_v2(input_shape=input_shape, depth=depth, num_classes=num_classes)
    else:
        model = myf.resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=myf.lr_schedule(0)),
                metrics=['accuracy'])
    model.summary()

    # Prepare model model saving directory.
    save_dir = os.path.join(RESULTS, 'checkpoints/restnet/')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
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


    ###### Train the RestNet
    print('Train the RestNest using real-time data augmentation.')
        
    model.fit_generator(train_data_gen,
                                steps_per_epoch=total_train,
                                epochs=epochs,
                                validation_data=val_data_gen,
                                validation_steps=total_val,
                                callbacks=callbacks)
          
    # Score trained model.
    scores = model.evaluate_generator(val_data_gen, verbose=1, steps=total_val)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('ERROR:\tPlease provide:\n1. GPU (yes = 1 / no = 0);\n2. the path of the project directory.\nUSAGE:\t%s USE_GPU PROJECT_DIR\n'%sys.argv[0])
        sys.exit(2)
    if not bool(int(sys.argv[1])):
        sys.stdout.write('Not using GPU.')
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
    