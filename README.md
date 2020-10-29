# Ensembles for Deep Learning-based Profiled Side-Channel Attacks
Repository code to support paper TCHES2020 (issue 4) paper "__Strength in Numbers: Improving Generalization with Ensembles in Machine Learning-based Profiled Side-channel Analysis__"

Link to the paper: https://tches.iacr.org/index.php/TCHES/article/view/8686

Authors: Guilherme Perin (Delft University of Technology, The Netherlands), Łukasz Chmielewski (Radboud University Nijmegen and Riscure BV) and Stjepan Picek (Delft University of Technology, The Netherlands)

## Datasets ##
The source code is prepared for three datasets: CHES CTF, ASCAD FIXED KEYS, ASCAD RANDOM KEYS.

### CHES CTF dataset ###
This dataset contains 45,000 profiling traces, with a fixed key, and additional 5,000 attacking traces with a different and fixed key. Each trace contains 2,200 samples that represent the processing of s-box operations around the first AES encryption round (Original CHES CTF webpage https://chesctf.riscure.com/2018/content?show=training provides only 10k traces for each device. The traces are already normalized with z-score normalization).

#### Download ####
CHES CTF dataset (ches_ctf.h5 file) can be downloaded from: https://www.dropbox.com/s/lpw1k3so99krmmq/ches_ctf.h5?dl=0

### ASCAD FIXED KEY dataset ###
Information about ASCAD FIXED KEY dataset can be found in the original github page: https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key


### ASCAD RANDOM KEY dataset ###
Information about ASCAD RANDOM KEY dataset can be found in the original github page: https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key

## Code ##

This section provides explanations on how to run the ensemble SCA code on the above datasets.

### Code execution ###

The ensemble code should be executed from __run_ensemble.py__ file. This file contain the following structure:

```python
ensemble_aes = EnsembleAES()
ensemble_aes.set_dataset("ches_ctf")  # "ascad_fixed_key", "ascad_random_key" or "ches_ctf"
ensemble_aes.set_leakage_model("HW")
ensemble_aes.set_target_byte(0)
ensemble_aes.set_mini_batch(400)
ensemble_aes.set_epochs(10)
ensemble_aes.run_ensemble(
    number_of_models=50,
    number_of_best_models=10
)
```

In the example above, the analysis will generate 50 models and create ensembles from the 50 models and from the 10 best models. After __run_ensemble__ method is finished, the user can plot guessing entropy and success rate for ensembles all models, ensembles best models, best validation model and best attack model. As an example, the following code can be used to generate the plot:

```python
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.plot(ensemble_aes.get_ge_best_model_validation(), label="GE best validation")
plt.plot(ensemble_aes.get_ge_best_model_attack(), label="GE best attack")
plt.plot(ensemble_aes.get_ge_ensemble(), label="GE Ensemble All Models")
plt.plot(ensemble_aes.get_ge_ensemble_best_models(), label="GE Ensemble Best Models")
plt.xlabel("Traces")
plt.ylabel("Guessing Entropy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(ensemble_aes.get_sr_best_model_validation(), label="SR best validation")
plt.plot(ensemble_aes.get_sr_best_model_attack(), label="SR best attack")
plt.plot(ensemble_aes.get_sr_ensemble(), label="SR Ensemble All Models")
plt.plot(ensemble_aes.get_sr_ensemble_best_models(), label="SR Ensemble Best Models")
plt.xlabel("Traces")
plt.ylabel("Success Rate")
plt.legend()
plt.show()
```

### Neural Networks ###
The provided code generate MLP or CNN with random hyperparameters according to certain user-defined ranges. In __commons/ensemble_aes.py__, the user can find the methods __run_mlp()__ and __run_cnn()__ for random MLPs and random CNNs, respectively. In __commons/neural_networks.py__ we provide the structure for the neural networks.
