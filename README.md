# Ensemble for Deep Learning-based Profiled Side-Channel Attacks
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
