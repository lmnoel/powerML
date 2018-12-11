# powerML
Logan Noel (lnoel) and Kirk Swanson (ks8)

## Requirements: 

Python packages: tensorflow, keras, hyperopt, numpy, pandas, matplotlib

Other: valgrind, and Python2.7, which needs to be installed from source for use with valgrind. See: https://stackoverflow.com/questions/20112989/how-to-use-valgrind-with-python


## Structure:

-optimizer.py: Uses MeasurePowerConsumption.powerMoniter and conducts a Bayesian search of the defined search space to find optimal model parameters.

-MeasurePowerConsumption.py: Uses runPythonScript. Invokes cachegrind and parses results to measure cache performance of a model. Uses a processor profile to accurately reflect miss penalty of the current system. 
 
-runModel.py: Load a model from file, and train it. Or, load a model from file, load the weights from file, and perform inference.

-Final Paper Multicol-Report on this project for CMSC 352.