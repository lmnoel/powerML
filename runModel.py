from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import sys
import json
import os


def load_config(config_file):
    try:
        with open(config_file, 'r') as jsonfile:
            configs = json.load(jsonfile)

        return configs
    except:
        raise Exception('Unable to load config_file')


def load_model_from_json(model_name):
    try:
        json_file = open(model_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        return model_from_json(loaded_model_json)
    except:
        raise Exception('Unable to load model file')


def train_model(model_name, data_name, config_name, weights_name, model_type):
    '''
    model_name is a json file.
    data_name is a .csv with traing data--two columns (x, y)
    and no headers (feel free to modify)
    weights_name is an .h5 file 
    config_name is a .json with parameters for fitting/evaluating
    '''

    model = load_model_from_json(model_name)
    configs = load_config(config_name)
    model.compile(loss=configs['loss'], optimizer=configs['optimizer'], metrics=['accuracy'])
    if model_type == 'dense_rectangle' or model_type == 'dense_triangle':

        if data_name == 'mnist':
            from keras.datasets import mnist
            (X, Y), _ = mnist.load_data()
            Y = to_categorical(Y, num_classes=10)

        elif data_name == 'mnist_small':
            from keras.datasets import mnist
            (X, Y), _ = mnist.load_data()
            indices = np.where(Y < 2)[0][:50]
            Y = Y[indices]
            X = X[indices, :, :]
            Y = to_categorical(Y, num_classes=2)

        model.fit(X, Y, epochs=configs['epochs'], batch_size=configs['batch_size'], verbose=1)
        
    if model_type == 'conv':

        if data_name == 'mnist':
            from keras.datasets import mnist
            (X, Y), _ = mnist.load_data()
            Y = to_categorical(Y, num_classes=10)
            X = X.reshape(60000, 28, 28, 1)

        elif data_name == 'mnist_small':
            from keras.datasets import mnist
            (X, Y), _ = mnist.load_data()
            indices = np.where(Y < 2)[0][:50]
            Y = Y[indices]
            X = X[indices, :, :]
            Y = to_categorical(Y, num_classes=2)
            X = X.reshape(50, 28, 28, 1)

        model.fit(X, Y, epochs=configs['epochs'], batch_size=configs['batch_size'], verbose=1)

    model.save_weights(weights_name)
    print('done training model')


def test_model(model_name, data_name, config_name, weights_name, model_type):
    '''
    model_name is a json file.
    data_name is a .csv with traing data--two columns (x, y)
    and no headers (feel free to modify)
    weights_name is an .h5 file 
    config_name is a .json with parameters for fitting/evaluating
    '''
    loaded_model = load_model_from_json(model_name)
    # load weights into new model
    loaded_model.load_weights(weights_name)

    configs = load_config(config_name)
    loaded_model.compile(loss=configs['loss'], optimizer=configs['optimizer'], metrics=['accuracy'])

    if model_type == 'dense_rectangle' or model_type == 'dense_triangle':

        if data_name == 'mnist':
            from keras.datasets import mnist
            _, (X, Y) = mnist.load_data()
            Y = to_categorical(Y, num_classes=10)

        elif data_name == 'mnist_small':
            from keras.datasets import mnist
            (X, Y), _ = mnist.load_data()
            indices = np.where(Y < 2)[0][:100]
            Y = Y[indices]
            X = X[indices, :, :]
            Y = to_categorical(Y, num_classes=2)
        score = loaded_model.evaluate(X, Y, verbose=1)

    if model_type == 'conv':

        if data_name == 'mnist':
            from keras.datasets import mnist
            _, (X, Y) = mnist.load_data()
            Y = to_categorical(Y, num_classes=10)
            X = X.reshape(10000, 28, 28, 1)

        elif data_name == 'mnist_small':
            from keras.datasets import mnist
            _, (X, Y) = mnist.load_data()
            indices = np.where(Y < 2)[0][:20]
            Y = Y[indices]
            X = X[indices, :, :]
            Y = to_categorical(Y, num_classes=2)
            X = X.reshape(20, 28, 28, 1)

        score = loaded_model.evaluate(X, Y, verbose=1)

    # save configs now with score to file
    configs['score'] = score

    try:
        os.remove(config_name)
    except:
        pass
    with open(config_name, 'w') as outfile:
        json.dump(configs, outfile)
    print('done testing model')

if __name__=="__main__":
    print('in main of runModel.py')
    print('sys.argv is:', sys.argv)
    if sys.argv[1] == 'train':
        assert(len(sys.argv) == 7), 'Incorrect number of arguments'
        train_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    elif sys.argv[1] == 'test':
        assert(len(sys.argv) == 7), 'Incorrect number of arguments'
        test_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

