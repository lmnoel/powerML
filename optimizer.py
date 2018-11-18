#Reference for loading/saving models: 
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/
from MeasurePowerConsumption import powerMoniter
from keras.models import Model
import json
import os

class searchSpace():
    def __init__(self, model_type):
        self.model_type = model_type

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

    def get_model(self):
        '''
        This method will return a keras model, does not need to
        be compiled (runModel.py will compile it).
        '''
        #TODO
        pass

    def get_configs(self):
        '''
        This function will return a dictionary containing parameters
        for the model (see runModel.py to observe how it will be used).
        '''
        #TODO
        pass


    def create_next_model_iteration(self, config_filename, model_filename):
        next_model = self.get_model()
        self.save_model_to_json(model_filename)
        next_configs = self.get_configs()
        self.save_configs_to_json(config_filename)


    def update_objectives(self, training_cost, inference_cost, model_score):
        '''
        This method will update the internal weights based on the previous
        training and inference cost. Inference cost is CPU cycles, so lower
        is better. Model score is accuracy.
        '''
        #TODO
        pass



class optimizer():
    def __init__(self):
        self.current_iteration = 0
        self.iterations = 20 #tbd
        self.power_moniter = powerMoniter()
        self.config_filename = 'configs.json'
        self.weights_filename = 'model_weights.h5'
        self.model_filename = 'model.json'
        

    def run(self, model_type, data_filename):
        search_space = search_space(model_type) 
        for current_iteration in range(iterations):
            search_space.create_next_model_iteration(self.config_filename, self.model_filename)
            training_cost = self.power_moniter.measure_training_efficiency(self.model_filename, 
                                                                           data_filename, 
                                                                           self.config_filename, 
                                                                           self.weights_filename)
            inference_cost, model_score= self.power_moniter.measure_inference_efficiency(model_filename, 
                                                                                         data_filename, 
                                                                                         config_filename, 
                                                                                         weights_filename)
            search_space.update_objectives(training_cost, inference_cost, model_score)
            self.log_record(iteration, training_cost, inference_cost, model_score)
        self.generate_report()

    def log_record(self, iteration, power, time, accuracy):
        
        pass

    def generate_report(self):
        #generate the 3d image
        pass

