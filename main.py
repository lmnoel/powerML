from optimizer import *
"""
Constructing an Optimizer object
model_type: Model types include 'dense' and 'conv'.  'dense' builds a fully-connected network, while 'conv' builds a 
network with convolutional layers.
data_filename: Data file names include 'mnist' and 'mnist_small'.  'mnist' is the full Keras mnist dataset, containing
60,0000 train examples and 10,000 test examples.  'mnist_small' contains only images corresponding to 0s and 1s, with
50 train examples and 50 test examples.  
cost: Boolean for measuring training and inference cost 

Running tests
Example: select a specific dense network architecture
test.run(iterations=1, num_layers=5, layer_widths=[3, 3, 3, 3, 3], batch_size=10)

Example: select a specific conv network architecture
test.run(iterations=2, num_layers=3, num_filters=[10, 5, 5], filter_sizes=[3, 3, 3], batch_size=10)

Example: conduct a random search
test.run(iterations=3)

Example: conduct a Bayesian optimization search
test.bayesian_opt(iterations=3)
"""


# Choose model type and dataset
test = Optimizer(model_type='dense_rectangle', data_filename='mnist_small', 
	epochs=2, cost=True, alpha=0.5, beta=0.0)

# Run tests
result = test.bayesian_opt(iterations=2)

# TODO
# Add epochs and valgrind as options to run() and bayesian_opt()