# Reference for loading/saving models:
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# Reference for Bayesian optimization package, hyperopt:
# http://hyperopt.github.io/hyperopt/
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
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


class SearchSpace:
    def __init__(self, model_type, data_filename):
        """
        Defines a fully-connected or convolutional neural network architecture and its hyperparameters.
        :param model_type: Choose a 'dense_rectangle' (fully-connected), 'dense_triangle' (fully connected) 
            or 'conv' (convolutional) network structure
        :param data_filename: Choose 'mnist' (full Keras version) or 'mnist_small' (just 0s and 1s) datasets
        """
        self.model_type = model_type
        self.data_filename = data_filename

    def get_model(self, num_layers=None, layer_widths=None, num_filters=None, filter_sizes=None):
        """
        Creates and saves a Keras model architecture.
        :param num_layers: Number of layers for a dense network (not including fixed output layer)
        :param layer_widths: Widths for each of the num_layers
        :param num_filters: List of number of filters for convolutional layers
        :param filter_sizes: List of square filter sizes for convolutional layers
        :return: Keras model
        """

        if self.model_type == 'dense_rectangle':

            if num_layers is None:
                # Select random number of layers
                max_layers = 10
                num_layers = np.random.randint(1, max_layers)

            if layer_widths is None:
                # Select random widths for the layers
                max_width = 20
                layer_widths = np.random.randint(1, max_width, num_layers)

            # Construct a dense network for the Keras MNIST dataset
            model = Sequential()
            model.add(Flatten(input_shape=(28, 28)))
            for layer_index in range(num_layers):
                model.add(Dense(layer_widths[layer_index], kernel_initializer='uniform', activation='relu'))
            if self.data_filename == 'mnist':
                model.add(Dense(10, kernel_initializer='uniform', activation='softmax'))
            # Output layer for data_filename='mnist_small', which only contains images of 0s and 1s
            elif self.data_filename == 'mnist_small':
                model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
            else:
                sys.exit('Choose mnist or mnist_small for data filename')
        elif self.model_type == 'dense_triangle':

            if num_layers is None:
                # Select random number of layers
                max_layers = 5
                num_layers = np.random.randint(1, max_layers)

            start_layer_width = 16

            # Construct a dense network for the Keras MNIST dataset
            model = Sequential()
            model.add(Flatten(input_shape=(28, 28)))
            for _ in range(num_layers):
                model.add(Dense(start_layer_width, kernel_initializer='uniform', activation='relu'))
                start_layer_width *= 2
            if self.data_filename == 'mnist':
                model.add(Dense(10, kernel_initializer='uniform', activation='softmax'))
            # Output layer for data_filename='mnist_small', which only contains images of 0s and 1s
            elif self.data_filename == 'mnist_small':
                model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
            else:
                sys.exit('Choose mnist or mnist_small for data filename')
        elif self.model_type == 'conv':

            if num_layers is None:
                # Select random number of convolutional layers
                max_layers = 5
                num_layers = np.random.randint(1, max_layers)

            if num_filters is None:
                # Select random number of filters for these layers
                max_filters = 5
                num_filters = np.random.randint(1, max_filters, num_layers)

            if filter_sizes is None:
                # Select random filter size for these layers
                max_filter_size = 10
                filter_sizes = np.random.randint(1, max_filter_size, num_layers)

            # Construct a convolutional network for the Keras MNIST dataset
            model = Sequential()
            for layer_index in range(num_layers):
                if layer_index == 0:
                    model.add(Conv2D(num_filters[layer_index], kernel_size=filter_sizes[layer_index], activation='relu',
                                     padding='same', input_shape=(28, 28, 1)))
                else:
                    model.add(Conv2D(num_filters[layer_index], kernel_size=filter_sizes[layer_index], activation='relu',
                                     padding='same'))
            model.add(Flatten())
            model.add(Dense(128, kernel_initializer='uniform', activation='relu'))
            if self.data_filename == 'mnist':
                model.add(Dense(10, kernel_initializer='uniform', activation='softmax'))
            elif self.data_filename == 'mnist_small':
                model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))

        else:
            sys.exit('Choose dense_rectangle, dense_triangle or conv for model_type')

        return model

    def get_configs(self, epochs, batch_size=None):
        """
        This function will return a dictionary containing hyperparameters for the model.
        :param epochs: Number of epochs for training
        :param batch_size: Batch size for training and testing
        :return: Dictionary of model hyperparameters
        """

        # Default batch size
        if batch_size is None:
            batch_size = 32

        if self.model_type == 'dense_rectangle':

            configs = {'epochs': epochs, 'batch_size': batch_size, 'optimizer': 'adam', 'loss': 'categorical_crossentropy'}
        elif self.model_type == 'dense_triangle':

            configs = {'epochs': epochs, 'batch_size': batch_size, 'optimizer': 'adam', 'loss': 'categorical_crossentropy'}
        elif self.model_type == 'conv':

            configs = {'epochs': epochs, 'batch_size': batch_size, 'optimizer': 'rmsprop',
                       'loss': 'categorical_crossentropy'}

        else:
            sys.exit('Choose dense or conv for model_type')

        return configs

    def create_next_model_iteration(self, config_filename, model_filename, epochs, num_layers=None, layer_widths=None,
                                    num_filters=None, filter_sizes=None, batch_size=None):
        """
        Save a model architecture and set of configurations to JSON files.
        :param config_filename: File name for configs JSON file
        :param model_filename: File name for model architecture JSON file
        :param epochs: Number of training epochs
        :param num_layers: Number of layers for a dense network (not including fixed output layer)
        :param layer_widths: Widths for each of the num_layers
        :param num_filters: List of number of filters for convolutional layers
        :param filter_sizes: List of square filter sizes for convolutional layers
        :param batch_size: Batch size for training and testing
        """
        next_model = self.get_model(num_layers=num_layers, layer_widths=layer_widths, num_filters=num_filters,
                                    filter_sizes=filter_sizes)
        save_model_to_json(model_filename, next_model)
        next_configs = self.get_configs(epochs, batch_size=batch_size)
        save_configs_to_json(config_filename, next_configs)


def save_configs_to_json(file_name, configs):
    """
    Saves configs file (hyperparameters) to JSON file.
    :param file_name: File name for JSON
    :param configs: Dictionary of hyperparameters
    """
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as fp:
        json.dump(configs, fp)


def save_model_to_json(file_name, model):
    """
    Saves model architecture to JSON file.
    :param file_name: File name for JSON
    :param model: Keras model
    """
    model_json = model.to_json()
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, "w") as json_file:
        json_file.write(model_json)


class Optimizer:
    def __init__(self, model_type, data_filename, cost=False, epochs=5, alpha=0.003, beta=0.002):
        """
        Class for testing accuracy, training cost (CPU cycles), and inference cost (CPU cycles) of different neural
        network architectures.
        :param model_type: Choose a 'dense_rectangle' or 'dense_triangle' (fully-connected) or 'conv' (convolutional) network structure
        :param data_filename: Choose 'mnist' (full Keras version) or 'mnist_small' (just 0s and 1s) datasets
        :param cost: Boolean for measuring training and inference cost
        :param epochs: Number of training epochs
        :param alpha: Coefficient for the weight of training CPU cost in objectuve
        :param beta: Coefficient for the weight of inference CPU cost in objective
        """
        self.power_monitor = powerMonitor()
        self.config_filename = 'configs.json'  # Filename for hyperparameters
        self.weights_filename = 'model_weights.h5'  # Filename for trained model weights
        self.model_filename = 'model.json'  # Filename for model architecture
        self.records = []  # Save records of training cost, inference cost, and model accuracy
        self.data_filename = data_filename  # Filename for training/test data
        self.iteration_index = 0  # Keep track of iterations from bayesian_opt function
        self.model_type = model_type  # Choose a 'dense_rectangle' or 'dense_triangle' (fully-connected) or 'conv' (convolutional) network structure
        self.architecture_file = open(get_architecture_file_name(), 'w')  # File name for recording model and parameters
        self.cost = cost
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        assert self.alpha >= 0 and self.beta >= 0, 'alpha and beta must be greater than 0.0'
        self.gamma = 1 - alpha - beta
        assert self.gamma >= 0, 'alpha + beta must be less than 1.0'

    def run(self, iterations, num_layers=None, layer_widths=None, num_filters=None, filter_sizes=None, batch_size=None):
        """
        Function for managing creation of network architectures and computing network costs and accuracies using either
        random search or a single fixed network architecture.  If num_layers and layer_widths are left as None, then
        get_model will choose them randomly at each iteration.  Otherwise, num_layers and layer_widths can be specified
        directly for a fixed architecture.  In this case, each iteration will compute cost and accuracy values for the
        same architecture.
        :param iterations: Number of times to run cost and accuracy computations
        :param num_layers: Number of layers for a dense network (not including fixed output layer)
        :param layer_widths: Widths for each of the num_layers
        :param num_filters: List of number of filters for convolutional layers
        :param filter_sizes: List of square filter sizes for convolutional layers
        :param batch_size: Batch size for training and testing
        """
        search_space = SearchSpace(self.model_type, self.data_filename)
        for current_iteration in range(iterations):
            search_space.create_next_model_iteration(self.config_filename, self.model_filename, self.epochs, num_layers,
                                                     layer_widths, num_filters, filter_sizes, batch_size)

            training_cost = self.power_monitor.measure_training_efficiency(self.model_filename,
                                                                           self.data_filename,
                                                                           self.config_filename,
                                                                           self.weights_filename,
                                                                           self.model_type,
                                                                           self.cost)

            inference_cost, model_score = self.power_monitor.measure_inference_efficiency(self.model_filename,
                                                                                          self.data_filename,
                                                                                          self.config_filename,
                                                                                          self.weights_filename,
                                                                                          self.model_type,
                                                                                          self.cost)

            self.log_record(current_iteration, training_cost, inference_cost, model_score)

            # Record the model architecture used for this iteration
            current_model = load_model_from_json(self.model_filename)
            self.architecture_file.write('Iteration ' + str(current_iteration) + '\n')
            current_model.summary(print_fn=lambda x: self.architecture_file.write(x + '\n'))
            self.architecture_file.write('\n')

            # Record the model hyperparameters used for this iteration
            with open(self.config_filename, 'r') as jsonfile:
                configs = json.load(jsonfile)
                self.architecture_file.write('batch size = {}'.format(configs['batch_size']) + '\n')
                self.architecture_file.write('epochs = {}'.format(configs['epochs']) + '\n')
                self.architecture_file.write('loss = {}'.format(configs['loss']) + '\n')
                self.architecture_file.write('optimizer = {}'.format(configs['optimizer']) + '\n')
                self.architecture_file.write('\n')

        self.architecture_file.close()

        # Generate a plot and csv record of costs and accuracies for all iterations
        self.generate_report()

    def bayesian_opt(self, iterations):
        """
        Function for managing a Bayesian hyperparameter optimization strategy.  We define a hyperparameter space to
        optimize over and choose an objective function to minimize.
        :param iterations: Number of times to run the optimization strategy
        :return: hyperopt's choice of best hyperparameters
        """

        max_layers = 10
        max_width = 20
        max_filters = 5
        max_filter_size = 10
        max_batchsize = 128

        if self.model_type == 'dense_rectangle':

            space = {
                'num_layers': 1 + hp.randint('num_layers', max_layers),
                'layer_widths': [1 + hp.randint('layer_widths_' + str(i), max_width) for i in range(max_layers)],
                'batch_size': 2 + hp.randint('batch_size', max_batchsize)
            }
        elif self.model_type == 'dense_triangle':

            space = {
                'num_layers': 1 + hp.randint('num_layers', max_layers),
                'batch_size': 2 + hp.randint('batch_size', max_batchsize)
            }
        elif self.model_type == 'conv':

            space = {
                'num_layers': 1 + hp.randint('num_layers', max_layers),
                'num_filters': [1 + hp.randint('num_filters_' + str(i), max_filters) for i in range(max_layers)],
                'filter_sizes': [1 + hp.randint('filter_sizes_' + str(i), max_filter_size) for i in range(max_layers)],
                'batch_size': 2 + hp.randint('batch_size', max_batchsize)
            }

        else:
            sys.exit('Choose dense_rectangle, dense_triangle or conv for model_type')

        # Run the optimization strategy.  tpe.suggest automatically chooses an appropriate algorithm for the
        # Bayesian optimization scheme.  fn is given the function that we want to minimize.
        tpe_best = fmin(fn=self.objective_function, space=space, algo=tpe.suggest, max_evals=iterations)

        self.architecture_file.close()

        # Generate a plot and csv record of costs and accuracies for all iterations
        self.generate_report()

        return 'Optimized architecture: ' + str(tpe_best)

    def objective_function(self, kwargs):
        """
        Objective function for Bayesian optimization strategy
        :param kwargs: Values from bayesian_opt "space" argument are passed here (e.g. num_layers)
        :return: The value to be minimized
        """
        search_space = SearchSpace(self.model_type, self.data_filename)
        search_space.create_next_model_iteration(self.config_filename, self.model_filename, self.epochs, **kwargs)

        training_cost = self.power_monitor.measure_training_efficiency(self.model_filename,
                                                                       self.data_filename,
                                                                       self.config_filename,
                                                                       self.weights_filename,
                                                                       self.model_type,
                                                                       self.cost)

        inference_cost, model_score = self.power_monitor.measure_inference_efficiency(self.model_filename,
                                                                                      self.data_filename,
                                                                                      self.config_filename,
                                                                                      self.weights_filename,
                                                                                      self.model_type,
                                                                                      self.cost)

        self.log_record(self.iteration_index, training_cost, inference_cost, model_score)

        # Record Keras model used for this iteration
        current_model = load_model_from_json(self.model_filename)
        self.architecture_file.write('Iteration ' + str(self.iteration_index) + '\n')
        print("Iteration: ", self.iteration_index)
        current_model.summary(print_fn=lambda x: self.architecture_file.write(x + '\n'))
        self.architecture_file.write('\n')

        # Record the model hyperparameters used for this iteration
        with open(self.config_filename, 'r') as jsonfile:
            configs = json.load(jsonfile)
            self.architecture_file.write('batch size = {}'.format(configs['batch_size']) + '\n')
            self.architecture_file.write('epochs = {}'.format(configs['epochs']) + '\n')
            self.architecture_file.write('loss = {}'.format(configs['loss']) + '\n')
            self.architecture_file.write('optimizer = {}'.format(configs['optimizer']) + '\n')
            self.architecture_file.write('\n')

        # Update iteration number
        self.iteration_index += 1

        # If costs are being computed, return a linear combination along with model_score (accuracy)
        # Normalization for cost is currently set to 15e9 based on typical values for the mnist_small dataset
        if self.iteration_index == 1:
            return -1.0 * model_score
        else:
            RECORDS_SCORE = 3
            RECORDS_TRAIN_COST = 1
            RECORDS_INFERENCE_COST = 2
            model_score_delta = self.records[-1][RECORDS_SCORE] - self.records[-2][RECORDS_SCORE]
            train_cost_delta = self.records[-1][RECORDS_TRAIN_COST] - self.records[-2][RECORDS_TRAIN_COST]
            inference_cost_delta = self.records[-1][RECORDS_INFERENCE_COST] - self.records[-2][RECORDS_INFERENCE_COST]
            return -1.0 * self.gamma * model_score_delta + train_cost_delta * self.alpha + inference_cost_delta * self.beta


    def log_record(self, iteration, training_cost, inference_cost, model_score):
        """
        Record information from training and testing a network.
        :param iteration: The number iteration this example corresponds to
        :param training_cost: CPU cycles for performing training
        :param inference_cost: CPU cycles for performing testing
        :param model_score: Accuracy of network on test set
        """
        self.records.append((iteration, training_cost, inference_cost, model_score))

    def generate_all_axes_fig(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        #ax = fig.add_subplot(111, projection='3d')
        for record in self.records:
            iteration, training_cost, inference_cost, model_score = record
            ax.scatter(iteration, inference_cost, model_score, c='r', marker='o', alpha=model_score ** 3)
            ax.scatter(iteration, training_cost, model_score, c='b', marker='o', alpha=model_score ** 3)

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Processor Cycles')
        ax.set_zlabel('Accuracy')
        ax.set_title('{} network, All Axes'.format(self.model_type))

        plt.savefig(get_fig_name("all_axes"))

    def generate_iteration_x_train_cost(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for record in self.records:
            iteration, training_cost, inference_cost, model_score = record
            ax.scatter(iteration, training_cost, c='b', marker='o')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Processor Cycles')
        ax.set_title('{} network, Iterations x Training Processor Cycles'.format(self.model_type))
        plt.savefig(get_fig_name("iteration_x_train_cost"))
    
    def generate_iteration_x_inference_cost(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for record in self.records:
            iteration, training_cost, inference_cost, model_score = record
            ax.scatter(iteration, inference_cost, c='r', marker='o')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Processor Cycles')
        ax.set_title('{} network, Iterations x Inference Processor Cycles'.format(self.model_type))
        plt.savefig(get_fig_name("iteration_x_inference_cost"))

    def generate_iteration_x_score(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for record in self.records:
            iteration, training_cost, inference_cost, model_score = record
            ax.scatter(iteration, model_score, c='k', marker='o')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Model Accuracy')
        ax.set_title('{} network, Iterations x Model Accuracy'.format(self.model_type))
        plt.savefig(get_fig_name("iteration_x_score"))

    def generate_report(self):
        """
        Save a 3d scatter plot to file.
        Iterations on x axis, training_cost (red) and inference_cost (blue) on y axis.
        And model score on z axis.
        """
        # From here: https://matplotlib.org/2.1.1/gallery/mplot3d/scatter3d.html
        # from mpl_toolkits.mplot3d import Axes3D
        self.generate_all_axes_fig()
        self.generate_iteration_x_score()
        self.generate_iteration_x_train_cost()
        self.generate_iteration_x_inference_cost()
        
        df = pd.DataFrame(self.records)
        df.to_csv(get_file_name())



def get_architecture_file_name():
    """
    Find appropriate name for the next architecture file (holds info on model architecture and parameters).
    :return: File name
    """
    counter = 0
    if not os.path.exists("architecture/"):
        os.mkdir("architecture/")
    file_name = 'architecture/architecture_0.txt'
    while os.path.isfile(file_name):
        counter += 1
        file_name = 'architecture/architecture_{}.txt'.format(counter)
    return file_name


def get_file_name():
    """
    Find appropriate name for the next report file (holds info on model costs and accuracy).
    :return: File name
    """
    counter = 0
    if not os.path.exists('reports/'):
        os.mkdir('reports/')
    file_name = 'reports/report_0.csv'
    while os.path.isfile(file_name):
        counter += 1
        file_name = 'reports/report_{}.csv'.format(counter)
    return file_name


def get_fig_name(keyword):
    """
    Find appropriate name for the next figure file.
    :return:
    """
    counter = 0
    if not os.path.exists("figures/"):
        os.mkdir("figures/")
    fig_name = 'figures/{}_figure_0.png'.format(keyword)
    while os.path.isfile(fig_name):
        counter += 1
        fig_name = 'figures/{}_figure_{}.png'.format(keyword, counter)
    return fig_name


def load_model_from_json(model_name):
    """
    Load a saved Keras model from a JSON file
    :param model_name: Name of JSON file containing saved Keras model architecture
    :return: Keras model
    """
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)
