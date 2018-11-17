from MeasurePowerConsumption import powerMoniter

class searchSpace():
    def __init__(self, model_type):
        self.model_type = model_type

    def get_next_parameter_set(self):
        #save a model to json
        #save parameter set to json
        pass

    def update_objectives(self, training_cost, inference_cost, model_score):
        pass



class optimizer():
    def __init__(self):
        self.current_iteration = 0
        self.iterations = 20 #tbd
        self.power_moniter = powerMoniter()
        

    def run(self, model_type, data_filename):
        config_filename = 'configs.json'
        weights_filename = 'model_weights.h5'
        search_space = search_space(model_type) 
        for current_iteration in range(iterations):
            search_space.get_next_parameter_set()
            training_cost = self.power_moniter.measure_training_efficiency(model_filename, data_filename, config_filename, weights_filename)
            inference_cost, model_score= self.power_moniter.measure_inference_efficiency(model_filename, data_filename, config_filename, weights_filename)
            search_space.update_objectives(training_cost, inference_cost, model_score)
            self.log_record(iteration, training_cost, inference_cost, model_score)
        self.generate_report()

    def log_record(self, iteration, power, time, accuracy):
        
        pass

    def generate_report(self):
        #generate the 3d image
        pass

