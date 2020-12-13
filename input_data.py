import os
import numpy as np

def load_data_label(base_model_dir):
    train_vec_filename = os.path.join(base_model_dir, "../data/AspecJ.train_vec.npy")
    train_label_filename = os.path.join(base_model_dir, '../data/AspecJ.train_label.npy')
    val_vec_filename = os.path.join(base_model_dir, "../data/AspecJ.val_vec.npy")
    val_label_filename = os.path.join(base_model_dir, '../data/AspecJ.val_label.npy')
    test_vec_filename = os.path.join(base_model_dir, '../data/AspecJ.test_vec.npy')
    test_label_filename = os.path.join(base_model_dir, '../data/AspecJ.test_label.npy')

    X_train = np.load(train_vec_filename)
    print('X_train', X_train.shape)
    Y_train = np.load(train_label_filename)
    print('Y_train', Y_train.shape)
    X_val = np.load(val_vec_filename)
    print('X_val', X_val.shape)
    Y_val = np.load(val_label_filename)
    print('Y_val', Y_val.shape)
    X_test = np.load(test_vec_filename)
    print('X_test', X_test.shape)
    Y_test = np.load(test_label_filename)
    print('Y_test', Y_test.shape)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_data_label_combine(X_train, X_test, X_val, Y_val, X1_train, X1_test):
    X_train_all = np.hstack((X_train, X1_train))
    X_val_all = np.hstack((X_val, Y_val))
    X_test_all = np.hstack((X_test, X1_test))
    return X_train_all, X_val_all, X_test_all