from keras.models import model_from_json
import numpy as np
import sys
import json

#Reference for loading/saving models: 
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

def load_config(config_file):
    return json.load(config_file)

def save_config(config_file):


def train_model(model_name, data_name, config_name, weights_name):
    '''
    model_name is a json file.
    data_name is a .csv with traing data--two columns (x, y)
    and no headers (feel free to modify)
    weights_name is an .h5 file 
    config_name is a .json with parameters for fitting/evaluating
    '''
    model = model_from_json(model_name)
    # load pima indians dataset
    dataset = np.loadtxt(data_name, delimiter=",")
    #adjust 8 to howver long the dataset is
    X = dataset[:,0:8]
    Y = dataset[:,8]
    configs = load_config(config_name)
   
    model.fit(X, Y, epochs=configs['epochs'], batch_size=configs['batch_size'], verbose=0)

    model.save_weights(weights_name)

def test_model(model_name, data_name, config_name, weights_name):
    '''
    model_name is a json file.
    data_name is a .csv with traing data--two columns (x, y)
    and no headers (feel free to modify)
    weights_name is an .h5 file 
    config_name is a .json with parameters for fitting/evaluating
    '''
    loaded_model = model_from_json(model_name)
    # load weights into new model
    loaded_model.load_weights(weights_name)
    configs = load_config(config_name)
    loaded_model.compile(loss=configs['loss'], optimizer=configs['optimizer'], metrics=['accuracy'])
    dataset = np.loadtxt(data_name, delimiter=",")
    #adjust 8 to howver long the dataset is
    X = dataset[:,0:8]
    Y = dataset[:,8]
    score = loaded_model.evaluate(X, Y, verbose=0)
    configs['score'] = score
    with open(config_name, 'w') as outfile:
        json.dump(configs, outfile)
    

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        print(train_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
    elif sys.argv[1] == 'test':
        print(test_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))

