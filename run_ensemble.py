from commons.ensemble_aes import EnsembleAES
import matplotlib.pyplot as plt

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ensemble_aes = EnsembleAES()
ensemble_aes.set_dataset("ascad_fixed_key")  # "ascad_fixed_key", "ascad_random_key" or "ches_ctf"
ensemble_aes.set_leakage_model("HW")
ensemble_aes.set_target_byte(2)
ensemble_aes.set_mini_batch(400)
ensemble_aes.set_epochs(10)
ensemble_aes.run_ensemble(
    number_of_models=5,
    number_of_best_models=3
)

# plotting GE and SR
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
