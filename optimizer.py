from MeasurePowerConsumption import powerMoniter

class searchSpace():
    def __init__(self, model_type):
        pass

    def get_next_parameter_set(self):
        pass

    def update_objectives(self, power):
        pass



class optimizer():
    def __init__(self):
        self.current_iteration = 0
        self.iterations = 20 #tbd
        self.power_moniter = powerMoniter()
        

    def run(self, model_type, data_filename):
        search_space = search_space(model_type) 
        for current_iteration in range(iterations):
            parameters = search_space.get_next_parameter_set()
            #update model with parameters
            #save model to file
            time, power, accuracy = self.powerMoniter.measure_consumption_of_model(model_filename, data_filename)
            search_space.update_objectives(power)
            self.log_record(iteration, time, power, accuracy)
        self.generate_report()

    def log_record(self, iteration, power, time, accuracy):
        #open up a csv and append row
        pass

    def generate_report(self):
        #generate the 3d image
        pass

