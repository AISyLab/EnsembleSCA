# Ensembles for Deep Learning-based Profiled Side-Channel Attacks
Repository code to support paper TCHES2020 (issue 4) paper "__Strength in Numbers: Improving Generalization with Ensembles in Machine Learning-based Profiled Side-channel Analysis__"

Link to the paper: https://tches.iacr.org/index.php/TCHES/article/view/8686

Authors: Guilherme Perin (Delft University of Technology, The Netherlands), Łukasz Chmielewski (Radboud University Nijmegen and Riscure BV) and Stjepan Picek (Delft University of Technology, The Netherlands)

## Datasets ##
The source code is prepared for three datasets: CHES CTF, ASCAD FIXED KEYS, ASCAD RANDOM KEYS.

### CHES CTF dataset ###
This dataset contains 45,000 profiling traces, with a fixed key, and additional 5,000 attacking traces with a different and fixed key. Each trace contains 2,200 samples that represent the processing of s-box operations around the first AES encryption round.

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

In the example above, the analysis will generate 50 models and create ensembles from the 50 models and from the 10 best models.
