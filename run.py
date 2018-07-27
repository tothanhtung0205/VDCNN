import os
import csv
import numpy as np
from keras.callbacks import ModelCheckpoint
from vdcnn import build_model
from vn_data_helper import *



def get_input_data(file_path,num_classes):
    a = data_helper()
    return a.load_txt_file(file_path,num_classes)


def train(input_file, num_classes, embedding_size, learning_rate, batch_size, num_epochs, save_dir=None, print_summary=False):
    # Stage 1: Convert raw texts into char-ids format && convert labels into one-hot vectors
    X_train, y_train = get_input_data(input_file,num_classes)

    # Stage 2: Build Model
    num_filters = [64, 128, 256, 512]

    model = build_model(num_filters=num_filters, num_classes=num_classes, embedding_size=embedding_size, learning_rate=learning_rate)

    # Stage 3: Training
    save_dir = save_dir if save_dir is not None else 'checkpoints'
    filepath = os.path.join(save_dir, "weights-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if print_summary:
        print(model.summary())

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=True
    )

train(
    input_file='dataset/train/',
    num_classes=7,
    embedding_size=16,
    learning_rate=0.001,
    batch_size=128,
    num_epochs=50,
    save_dir="model/"

)