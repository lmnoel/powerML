from optimizer import *
test = optimizer(data_filename='mnist')
# check = test.objective_function(model_type='dense', num_layers=1, layer_widths=[5], data_filename='dense_dataset.csv')
# print(check)
check = test.bayesian_opt(model_type='dense')
