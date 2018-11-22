from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
import json
import os

#Reference for loading/saving models: 
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

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

    print('check')

    model = load_model_from_json(model_name)
    configs = load_config(config_name)
    model.compile(loss=configs['loss'], optimizer=configs['optimizer'], metrics=['accuracy'])
    
    if model_type == 'dense':

        
        # load pima indians dataset
        dataset = np.loadtxt(data_name, delimiter=",")
        #adjust 8 to howver long the dataset is
        X = dataset[:,0:8]
        Y = dataset[:,8]
        
    if model_type == 'conv':


        train_datagen = ImageDataGenerator(
            data_format='channels_last',
            vertical_flip=True,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            data_name+'/Train',
            target_size=(224,224),
            batch_size=configs['batch_size'],
            class_mode='categorical')

        history = model.fit_generator(
            train_generator,
            steps_per_epoch = 15,
            epochs = configs['epochs'],
            shuffle = True,
            workers = 4,
            verbose=1
            )

    model.save_weights(weights_name)

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

    if model_type == 'dense':
        
        dataset = np.loadtxt(data_name, delimiter=",")
        #adjust 8 to howver long the dataset is
        X = dataset[:,0:8]
        Y = dataset[:,8]
        score = loaded_model.evaluate(X, Y, verbose=0)

    if model_type == 'conv':

        test_datagen = ImageDataGenerator(
            data_format='channels_last')

        test_generator = test_datagen.flow_from_directory(
            data_name+'/Test',
            target_size=(224,224),
            batch_size=configs['batch_size'],
            class_mode='categorical')

        score = loaded_model.evaluate_generator(generator=test_generator, steps=len(test_generator))
        score = [score[1], score[0]]

    #save configs now with score to file
    configs['score'] = score

    try:
        os.remove(config_name)
    except:
        pass
    with open(config_name, 'w') as outfile:
        json.dump(configs, outfile)

    
if sys.argv[1] == 'train':
    assert(len(sys.argv) == 7), 'Incorrect number of arguments'
    train_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
elif sys.argv[1] == 'test':
    assert(len(sys.argv) == 7), 'Incorrect number of arguments'
    test_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

