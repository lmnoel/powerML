import subprocess
import os
import json
#https://www.7-cpu.com/cpu/Broadwell.html
class powerMoniter():
    processor_profiles = {'core_i5_broadwell_2_7ghz':{'l1d':4.5,'l2':12, 'l3':38, 'memory':38}}
    def __init__(self, processorProfile='core_i5_broadwell_2_7ghz'):
        self.cache_miss_weights = powerMoniter.processor_profiles[processorProfile]

    @staticmethod
    def parse_number_from_line(line, skip_parens):
        i = len(line) - 1
        if skip_parens:
            while line[i] != '(':
                i -= 1
            i -= 1
            while line[i] == ' ':
                i -= 1
        begin_of_parens = i
        while (line[i] != ' '):
            i -= 1
        if skip_parens:
            number_string =line[i + 1:begin_of_parens + 1].replace(',', '')
        else:
            number_string = line[i + 1:].replace(',', '')
        return int(number_string)

    def return_weighted_cycles(self, output):
        l1_cache_accesses, l3_cache_accesses, memory_accesses = None, None, None
        for line in output.split('\n'):
            if 'D   refs' in line:
                l1_cache_accesses = self.parse_number_from_line(line, skip_parens=True)
            elif 'D1  misses' in line:

                l3_cache_accesses = self.parse_number_from_line(line, skip_parens=True)
            elif 'LLd misses' in line:
                memory_accesses = self.parse_number_from_line(line, skip_parens=True)
                break

        assert l1_cache_accesses and l3_cache_accesses and memory_accesses, 'Failed to match to cache data'
        return int(l1_cache_accesses * self.cache_miss_weights['l1d'] + 
               l3_cache_accesses * self.cache_miss_weights['l3'] +
               memory_accesses * self.cache_miss_weights['memory'])


    def measure_training_efficiency(self, model_file, data_file, config_file, weights_file):
        return self.measure_model_efficiency('train', model_file, data_file, config_file, weights_file)
        
    def measure_inference_efficiency(self, model_file, data_file, config_file, weights_file):
        cost =  self.measure_model_efficiency('test', model_file, data_file, config_file, weights_file)
        try:
            with open(config_file, 'r') as jsonfile:
                configs = json.load(jsonfile)
                score = configs['score'][0]
        except:
            raise Exception('Unable to load config file to read score of model')
        return cost, score

    def measure_model_efficiency(self, type_, model_file, data_file, config_file, weights_file):

        log_file_name = 'temp_log'
        try:
            subpoutput = subprocess.check_output(['valgrind', '--tool=cachegrind',
                                            '--log-file={}'.format(log_file_name),
                                              './runPythonScript', 'runModel.py' , type_,
                                              model_file, data_file, config_file, weights_file, ' >', log_file_name])

        except:
            raise Exception('failed to run cachegrind')
        try:
            with open(log_file_name, 'r') as file:
                output = file.read()
            os.remove(log_file_name)
        except:
            print('unable to load log file')
        
        try:
            return self.return_weighted_cycles(output)
        except:
            raise Exception('processorProfile is invalid')