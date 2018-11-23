# Reference for loading/saving models:
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from MeasurePowerConsumption import powerMonitor
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import json
import os
import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin


class searchSpace():
    def __init__(self, model_type):
        self.model_type = model_type
        self.records = []

    @staticmethod
    def save_model_to_json(file_name, model):
        model_json = model.to_json()
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, "w") as json_file:
            json_file.write(model_json)

    @staticmethod
    def save_configs_to_json(file_name, configs):
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'w') as fp:
            json.dump(configs, fp)

    def get_model(self, num_layers=None, layer_widths=None):
        '''
        This method will return a keras model, does not need to
        be compiled (runModel.py will compile it).
        '''

        if self.model_type == 'dense':

            if num_layers is None:
                # Select random number of layers
                num_layers = np.random.randint(1, 10)

            if layer_widths is None:
                # Select random widths for these layers
                layer_widths = np.random.randint(1, 20, num_layers)

            # Construct a dense network
            model = Sequential()
            model.add(Flatten(input_shape=(28, 28)))
            for layer_index in range(num_layers):
                if layer_index == 0:
                    model.add(
                        Dense(layer_widths[layer_index], input_dim=8, kernel_initializer='uniform', activation='relu'))
                else:
                    model.add(Dense(layer_widths[layer_index], kernel_initializer='uniform', activation='relu'))
            model.add(Dense(10, kernel_initializer='uniform', activation='softmax'))

        if self.model_type == 'conv':

            # Select random number of convolutional layers
            num_layers = np.random.randint(1, 10)

            # Select random number of filters for these layers
            num_filters = np.random.randint(1, 5, num_layers)

            # Select random filter size for these layers
            filter_sizes = np.random.randint(1, 10, num_layers)

            model = Sequential()
            for layer_index in range(num_layers):
                if layer_index == 0:
                    model.add(
                        Conv2D(num_filters[layer_index], filter_sizes[layer_index], strides=(1, 1), padding='valid',
                               data_format='channels_last', input_shape=(224, 224, 3)))
                else:
                    model.add(
                        Conv2D(num_filters[layer_index], filter_sizes[layer_index], strides=(1, 1), padding='valid',
                               data_format='channels_last'))
            model.add(Flatten())
            model.add(Dense(10, activation='relu'))
            model.add(Dense(2, activation='softmax'))

        return model

    def get_configs(self):
        '''
        This function will return a dictionary containing parameters
        for the model (see runModel.py to observe how it will be used).
        '''
        if self.model_type == 'dense':
            configs = {'epochs': 5, 'batch_size': 32, 'optimizer': 'adam', 'loss': 'categorical_crossentropy'}

        elif self.model_type == 'conv':
            configs = {'epochs': 1, 'batch_size': 10, 'optimizer': 'rmsprop', 'loss': 'categorical_crossentropy'}

        return configs

    def create_next_model_iteration(self, config_filename, model_filename, num_layers=None, layer_widths=None):
        next_model = self.get_model(num_layers=num_layers, layer_widths=layer_widths)
        self.save_model_to_json(model_filename, next_model)
        next_configs = self.get_configs()
        self.save_configs_to_json(config_filename, next_configs)

    def update_objectives(self, training_cost, inference_cost, model_score):
        '''
        This method will update the internal weights based on the previous
        training and inference cost. Inference cost is CPU cycles, so lower
        is better. Model score is accuracy.
        '''
        # TODO
        pass

    def log_record(self, iteration, training_cost, inference_cost, model_score):
        self.records.append((iteration, training_cost, inference_cost, model_score))


class optimizer():
    def __init__(self, data_filename):
        self.power_monitor = powerMonitor()
        self.config_filename = 'configs.json'
        self.weights_filename = 'model_weights.h5'
        self.model_filename = 'model.json'
        self.records = []
        self.data_filename = data_filename

    def run(self, model_type, iterations):
        search_space = searchSpace(model_type)
        architecture_file = open(self.get_architecture_file_name(), 'w')
        for current_iteration in range(iterations):
            search_space.create_next_model_iteration(self.config_filename, self.model_filename)

            training_cost = self.power_monitor.measure_training_efficiency(self.model_filename,
                                                                           self.data_filename,
                                                                           self.config_filename,
                                                                           self.weights_filename,
                                                                           model_type)

            inference_cost, model_score = self.power_monitor.measure_inference_efficiency(self.model_filename,
                                                                                          self.data_filename,
                                                                                          self.config_filename,
                                                                                          self.weights_filename,
                                                                                          model_type)

            print(training_cost, inference_cost, model_score)

            # search_space.log_record(current_iteration, training_cost, inference_cost, model_score)
            # search_space.update_objectives(training_cost, inference_cost, model_score)
            self.log_record(current_iteration, training_cost, inference_cost, model_score)

            # Record Keras model used for this iteration
            current_model = self.load_model_from_json(self.model_filename)
            architecture_file.write('Iteration ' + str(current_iteration) + '\n')
            current_model.summary(print_fn=lambda x: architecture_file.write(x + '\n'))
            architecture_file.write('\n')
        architecture_file.close()

        self.generate_report()

    def bayesian_opt(self, model_type):

        space = {
            'num_layers': 1 + hp.randint('num_layers', 5),
            'layer_widths': [1 + hp.randint('layer_widths_'+str(i), 10) for i in range(6)]
        }

        tpe_best = fmin(fn=self.objective_function, space=space,
                        algo=tpe.suggest, max_evals=3)

        print(tpe_best)

    def objective_function(self, kwargs, model_type='dense'):
        search_space = searchSpace(model_type)
        architecture_file = open(self.get_architecture_file_name(), 'w')
        search_space.create_next_model_iteration(self.config_filename, self.model_filename, **kwargs)

        training_cost = self.power_monitor.measure_training_efficiency(self.model_filename,
                                                                       self.data_filename,
                                                                       self.config_filename,
                                                                       self.weights_filename,
                                                                       model_type)

        inference_cost, model_score = self.power_monitor.measure_inference_efficiency(self.model_filename,
                                                                                      self.data_filename,
                                                                                      self.config_filename,
                                                                                      self.weights_filename,
                                                                                      model_type)

        print(kwargs, model_score)

        return -1.0*model_score

    def load_model_from_json(self, model_name):
        try:
            json_file = open(model_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            return model_from_json(loaded_model_json)
        except:
            raise Exception('Unable to load model file')

    def log_record(self, iteration, training_cost, inference_cost, model_score):
        self.records.append((iteration, training_cost, inference_cost, model_score))

    def get_fig_name(self):
        counter = 0
        fig_name = 'figure_0.png'
        while os.path.isfile(fig_name):
            counter += 1
            fig_name = 'figure_{}.png'.format(counter)
        return fig_name

    def get_file_name(self):
        counter = 0
        file_name = 'report_0.csv'
        while os.path.isfile(file_name):
            counter += 1
            file_name = 'report_{}.csv'.format(counter)
        return file_name

    def get_architecture_file_name(self):
        counter = 0
        file_name = 'architecture_0.txt'
        while os.path.isfile(file_name):
            counter += 1
            file_name = 'architecture_{}.txt'.format(counter)
        return file_name

    def generate_report(self):
        '''
        Save a 3d scatter plot to file.
        Iterations on x axis, training_cost (red) and inference_cost (blue) on y axis.
        And model score on z axis.
        '''
        # From here: https://matplotlib.org/2.1.1/gallery/mplot3d/scatter3d.html
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for record in self.records:
        #     iteration, training_cost, inference_cost, model_score = record
        #     ax.scatter(iteration, inference_cost, model_score, c='r', marker='o', alpha=model_score ** 3)
        #     ax.scatter(iteration, training_cost, model_score, c='b', marker='o', alpha=model_score ** 3)

        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Cost--inference in red, training in blue \n(Processor Cycles)')
        # ax.set_zlabel('Accuracy')

        # plt.savefig(self.get_fig_name())
        df = pd.DataFrame(self.records)
        df.to_csv(self.get_file_name())

        #
