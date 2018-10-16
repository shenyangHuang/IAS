# Exploring Class-incremental Learning Using Incremental Architecture Search

# To run the experiments:

**To replicate the MNIST experiment in the paper:**
```
python -u runMLP_SAS.py |& tee output.txt
```

**To run MNIST experiment with a fixed architecture:**
```
python -u fixed_architecture.py |& tee output.txt
```

**To replicate the incremental experiment of Fashion-MNIST and MNIST in the paper**
```
python -u run_Incremental_SAS.py |& tee output.txt
```

Note that both experiments utilizes around 8 CPUs and might take long to run.
output.txt reports the validation accuracy per epoch as well as the test accuracies of each class once training is finished

# Dependencies:
Python 3.6.4

Keras 2.2.0

Tensorflow 1.8.0 (cpu version)

numpy 1.14.5

pathos 0.2.1


