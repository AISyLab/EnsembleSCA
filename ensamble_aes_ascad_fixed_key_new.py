from tensorflow.keras import backend as backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Conv1D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import numpy as np
import random
import h5py
import matplotlib.pyplot as plt
import os.path
import sys
from sklearn import preprocessing


def aes_sbox_table():
    return np.array(
        [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76, 0xCA, 0x82,
         0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26,
         0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96,
         0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
         0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB,
         0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F,
         0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF,
         0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
         0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32,
         0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
         0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6,
         0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
         0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E,
         0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF, 0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F,
         0xB0, 0x54, 0xBB, 0x16])


def aes_labelize(trace_data, leakage_model, byte, param, key=None):
    pt_ct = [row[byte + param["input_offset"]] for row in trace_data]

    if key is not None:
        key_byte = np.full(len(pt_ct), key[byte])
    else:
        key_byte = np.full(len(pt_ct), bytearray.fromhex(param["key"])[byte])

    state = [int(x) ^ int(k) for x, k in zip(np.asarray(pt_ct[:]), key_byte)]

    intermediate_values = aes_sbox_table()[state]

    if leakage_model == "HW":
        return [bin(iv).count("1") for iv in intermediate_values]
    else:
        return intermediate_values


def __add_if_one(value):
    return 1 if value == 1 else 0


def ge_and_sr(runs, model, param, leakage_model, byte, x_test, test_trace_data, step, fraction):
    nt = len(x_test)
    nt_kr = int(nt / fraction)
    nt_interval = int(nt / (step * fraction))
    key_ranking_sum = np.zeros(nt_interval)
    success_rate_sum = np.zeros(nt_interval)
    key_probabilities_key_ranks = np.zeros((runs, nt, 256))

    # ---------------------------------------------------------------------------------------------------------#
    # compute labels for all key hypothesis
    # ---------------------------------------------------------------------------------------------------------#
    labels_key_hypothesis = np.zeros((256, nt))
    for key_byte_hypothesis in range(0, 256):
        key_h = bytearray.fromhex(param["key"])
        key_h[byte] = key_byte_hypothesis
        labels_key_hypothesis[key_byte_hypothesis][:] = aes_labelize(test_trace_data, leakage_model, byte, param, key=key_h)

    # ---------------------------------------------------------------------------------------------------------#
    # predict output probabilities for shuffled test or validation set
    # ---------------------------------------------------------------------------------------------------------#
    output_probabilities = model.predict(x_test)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in labels_key_hypothesis[:]])
        ]

    for run in range(runs):

        probabilities_kg_all_traces_shuffled = shuffle(probabilities_kg_all_traces, random_state=random.randint(0, 100000))
        key_probabilities = np.zeros(256)
        kr_count = 0
        for index in range(nt_kr):
            key_probabilities += np.log(probabilities_kg_all_traces_shuffled[index] + 1e-36)
            key_probabilities_key_ranks[run][index] = probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
            if (index + 1) % step == 0:
                key_ranking_good_key = list(key_probabilities_sorted).index(param["good_key"]) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                if key_ranking_good_key == 1:
                    success_rate_sum[kr_count] += 1
                kr_count += 1
        print(
            "KR: {} | GE for correct key ({}): {})".format(run, param["good_key"], key_ranking_sum[nt_interval - 1] / (run + 1)))

    guessing_entropy = key_ranking_sum / runs
    success_rate = success_rate_sum / runs

    hf = h5py.File('output_prob.h5', 'w')
    hf.create_dataset('output_probabilities', data=output_probabilities)
    hf.close()

    return guessing_entropy, success_rate, key_probabilities_key_ranks


def get_best_models(n_models, result_models_validation, n_traces):
    result_number_of_traces_val = []
    for model_index in range(n_models):
        if result_models_validation[model_index][n_traces - 1] == 1:
            for index in range(n_traces - 1, -1, -1):
                if result_models_validation[model_index][index] != 1:
                    result_number_of_traces_val.append(
                        [result_models_validation[model_index][n_traces - 1], index + 1,
                         model_index])
                    break
        else:
            result_number_of_traces_val.append(
                [result_models_validation[model_index][n_traces - 1], n_traces,
                 model_index])

    sorted_models = sorted(result_number_of_traces_val, key=lambda l: l[:])

    list_of_best_models = []
    for model_index in range(n_models):
        list_of_best_models.append(sorted_models[model_index][2])

    return list_of_best_models


def mlp_random(classes, number_of_samples, activation, neurons, layers, learning_rate):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(number_of_samples,)))
    for l_i in range(layers):
        model.add(Dense(neurons, activation=activation, kernel_initializer='he_uniform', bias_initializer='zeros'))
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def cnn_random(classes, number_of_samples, activation, neurons, conv_layers, filters, kernel_size, stride, layers, learning_rate):
    model = Sequential()
    for layer_index in range(conv_layers):
        if layer_index == 0:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='valid',
                             input_shape=(number_of_samples, 1)))
        else:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation='relu', padding='valid'))

    model.add(Flatten())
    for layer_index in range(layers):
        model.add(Dense(neurons, activation=activation, kernel_initializer='random_uniform', bias_initializer='zeros'))

    model.add(Dense(classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    return model


def run_mlp(x_t, y_t, x_v, y_v, params, leakage_model, byte, x_test_1, test_data_1, x_test_2, test_data_2, step, fraction):
    mini_batch = random.randrange(500, 1000, 100)
    learning_rate = random.uniform(0.0001, 0.001)
    activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
    layers = random.randrange(2, 8, 1)
    neurons = random.randrange(500, 800, 100)

    model = mlp_random(classes, params["number_of_samples"], activation, neurons, layers, learning_rate)
    model.fit(
        x=x_t,
        y=y_t,
        batch_size=mini_batch,
        verbose=1,
        epochs=25,
        shuffle=True,
        validation_data=(x_v, y_v),
        callbacks=[])

    ge_test_1, sr_model, kp_krs = ge_and_sr(100, model, params, leakage_model, byte, x_test_1, test_data_1, step, fraction)
    ge_test_2, _, _ = ge_and_sr(100, model, params, leakage_model, byte, x_test_2, test_data_2, step, fraction)

    backend.clear_session()

    return ge_test_1, ge_test_2, sr_model, kp_krs


def run_cnn(x_t, y_t, x_v, y_v, params, leakage_model, byte, x_test_1, test_data_1, x_test_2, test_data_2, step, fraction):
    mini_batch = random.randrange(500, 1000, 100)
    learning_rate = random.uniform(0.0001, 0.001)
    activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
    dense_layers = random.randrange(2, 8, 1)
    neurons = random.randrange(500, 800, 100)
    conv_layers = random.randrange(1, 2, 1)
    filters = random.randrange(8, 32, 4)
    kernel_size = random.randrange(10, 20, 2)
    stride = random.randrange(5, 10, 5)

    model = cnn_random(classes, target_params["number_of_samples"], activation, neurons, conv_layers, filters, kernel_size, stride,
                       dense_layers, learning_rate)
    model.fit(
        x=x_t,
        y=y_t,
        batch_size=mini_batch,
        verbose=1,
        epochs=25,
        shuffle=True,
        validation_data=(x_v, y_v),
        callbacks=[])

    ge_test_1, sr_model, kp_krs = ge_and_sr(100, model, params, leakage_model, byte, x_test_1, test_data_1, step, fraction)
    ge_test_2, _, _ = ge_and_sr(100, model, params, leakage_model, byte, x_test_2, test_data_2, step, fraction)

    backend.clear_session()

    return ge_test_1, ge_test_2, sr_model, kp_krs


def check_file_exists(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return


#### ASCAD helper to load profiling and attack data (traces and labels) (source : https://github.com/ANSSI-FR/ASCAD)
# Loads the profiling and attack datasets from the ASCAD database
def load_ascad(ascad_database_file, load_metadata=False):
    check_file_exists(ascad_database_file)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
        sys.exit(-1)
    # Load profiling traces
    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    # Load profiling labels
    Y_profiling = np.array(in_file['Profiling_traces/labels'])
    # Load attacking traces
    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    # Load attacking labels
    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling, Y_profiling), (X_attack, Y_attack), (
        in_file['Profiling_traces/metadata']['plaintext'], in_file['Attack_traces/metadata']['plaintext'])

def create_z_score_norm(dataset):
    z_score_mean = np.mean(dataset, axis=0)
    z_score_std = np.std(dataset, axis=0)
    return z_score_mean, z_score_std


def apply_z_score_norm(dataset, z_score_mean, z_score_std):
    for index in range(len(dataset)):
        dataset[index] = (dataset[index] - z_score_mean) / z_score_std

if __name__ == "__main__":

    parameters_ascad_fixed_key = {
        "key": "4DFBE0F27221FE10A78D4ADC8E490469",
        "key_offset": 16,
        "input_offset": 0,
        "data_length": 50,
        "good_key": 224,
        "random_key": True,
        "number_of_samples": 700
    }

    target_params = parameters_ascad_fixed_key

    l_model = "HW"
    classes = 9
    target_byte = 2

    n_train = 49000
    n_val = 1000
    n_test = 5000

    ASCAD_data_folder = "D:/traces/ASCAD_data/ASCAD_data/ASCAD_databases/"

    # Load the profiling traces
    (X_profiling, _), (X_attack, _), (plt_profiling, plt_attack) = load_ascad(ASCAD_data_folder + "ASCAD.h5",
                                                                                               load_metadata=True)

    # normalize with z-score
    z_score_mean, z_score_std = create_z_score_norm(X_profiling)
    apply_z_score_norm(X_profiling, z_score_mean, z_score_std)
    apply_z_score_norm(X_attack, z_score_mean, z_score_std)

    # labelize according to leakage model
    train_labels = aes_labelize(plt_profiling[0:n_train], l_model, target_byte, target_params)
    validation_labels = aes_labelize(plt_profiling[n_train:n_train + n_val], l_model, target_byte, target_params)
    test_labels_1 = aes_labelize(plt_attack[0: n_test], l_model, target_byte, target_params)
    test_labels_2 = aes_labelize(plt_attack[n_test: 2*n_test], l_model, target_byte, target_params)

    # convert labels to categorical labels
    Y_profiling = to_categorical(train_labels, num_classes=classes)
    Y_validation = to_categorical(validation_labels, num_classes=classes)
    y_test_1 = to_categorical(test_labels_1, num_classes=classes)
    y_test_2 = to_categorical(test_labels_2, num_classes=classes)

    Y_data_1 = plt_attack[0: n_test]
    Y_data_2 = plt_attack[n_test: 2*n_test]

    X_profiling = X_profiling.astype('float32')
    X_attack = X_attack.astype('float32')

    # split sets
    X_train = X_profiling[0:n_train]
    X_validation = X_profiling[n_train:n_train + n_val]
    X_test_1 = X_attack[0:n_test]
    X_test_2 = X_attack[n_test:2*n_test]

    kr_step = 1
    kr_fraction = 10
    kr_runs = 100

    ge_all_set_1 = []
    sr_all_set_1 = []
    ge_all_set_2 = []
    k_ps_all = []

    number_of_models = 5
    number_of_best_models = 3
    kr_nt = int(len(X_test_1) / (kr_step * kr_fraction))

    # train random MLP
    # for model_index in range(number_of_models):
    #     ge_test_1, ge_test_2, sr_model, kp_krs = run_mlp(X_train, Y_profiling, X_validation, Y_validation,
    #                                                      target_params, l_model, target_byte,
    #                                                      X_test_1, Y_data_1,
    #                                                      X_test_2, Y_data_2,
    #                                                      kr_step, kr_fraction)
    #     ge_all_set_1.append(ge_test_1)
    #     ge_all_set_2.append(ge_test_2)
    #     sr_all_set_1.append(sr_model)
    #     k_ps_all.append(kp_krs)

    # train random CNN

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_validation_reshaped = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
    x_test_1_reshaped = X_test_1.reshape((X_test_1.shape[0], X_test_1.shape[1], 1))
    x_test_2_reshaped = X_test_2.reshape((X_test_2.shape[0], X_test_2.shape[1], 1))
    for model_index in range(number_of_models):
        ge_test_1, ge_test_2, sr_model, kp_krs = run_cnn(X_train_reshaped, Y_profiling, X_validation_reshaped, Y_validation,
                                                         target_params, l_model, target_byte,
                                                         x_test_1_reshaped, Y_data_1,
                                                         x_test_2_reshaped, Y_data_2,
                                                         kr_step, kr_fraction)
        ge_all_set_1.append(ge_test_1)
        ge_all_set_2.append(ge_test_2)
        sr_all_set_1.append(sr_model)
        k_ps_all.append(kp_krs)

    # Ensemble
    list_of_best_models = get_best_models(number_of_models, ge_all_set_2, kr_nt)

    kr_ensemble = np.zeros(kr_nt)
    krs_ensemble = np.zeros((kr_runs, kr_nt))
    kr_ensemble_best_models = np.zeros(kr_nt)
    krs_ensemble_best_models = np.zeros((kr_runs, kr_nt))

    for run in range(kr_runs):

        key_p_ensemble = np.zeros(256)
        key_p_ensemble_best_models = np.zeros(256)

        for index in range(kr_nt):
            for model_index in range(number_of_models):
                key_p_ensemble += np.log(k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)
            for model_index in range(number_of_best_models):
                key_p_ensemble_best_models += np.log(k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)

            key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]
            key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]

            kr_position = list(key_p_ensemble_sorted).index(target_params["good_key"]) + 1
            kr_ensemble[index] += kr_position
            krs_ensemble[run][index] = kr_position

            kr_ensemble_best_models[index] += list(key_p_ensemble_best_models_sorted).index(target_params["good_key"]) + 1
            krs_ensemble_best_models[run][index] = list(key_p_ensemble_best_models_sorted).index(target_params["good_key"]) + 1

        print("Run {} - GE {} models: {} | GE {} models: {} | ".format(run, number_of_models,
                                                                       int(kr_ensemble[kr_nt - 1] / (run + 1)),
                                                                       number_of_best_models,
                                                                       int(kr_ensemble_best_models[kr_nt - 1] / (run + 1))))

    ge_ensemble = kr_ensemble / kr_runs
    ge_ensemble_best_models = kr_ensemble_best_models / kr_runs

    sr_ensemble = np.zeros(kr_nt)
    sr_ensemble_best_models = np.zeros(kr_nt)

    sr_best_model_set_1 = np.zeros(kr_nt)
    sr_best_model_set_2 = np.zeros(kr_nt)

    for index in range(kr_nt):
        for run in range(kr_runs):
            sr_ensemble[index] += __add_if_one(krs_ensemble[run][index])
            sr_ensemble_best_models[index] += __add_if_one(krs_ensemble_best_models[run][index])
        sr_best_model_set_1[index] += __add_if_one(ge_all_set_1[list_of_best_models[0]][index])
        sr_best_model_set_2[index] += __add_if_one(ge_all_set_2[list_of_best_models[0]][index])

    plt.subplot(1, 2, 1)
    plt.plot(ge_all_set_1[list_of_best_models[0]], label="GE best set 1")
    plt.plot(ge_all_set_2[list_of_best_models[0]], label="GE best set 2")
    plt.plot(ge_ensemble, label="GE Ensemble All Models")
    plt.plot(ge_ensemble_best_models, label="GE Ensemble Best Models")
    plt.xlabel("Traces")
    plt.ylabel("Guessing Entropy")
    plt.xlim([0, kr_nt])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(sr_best_model_set_1, label="SR best set 1")
    plt.plot(sr_best_model_set_2, label="SR best set 2")
    plt.plot(sr_ensemble, label="SR Ensemble All Models")
    plt.plot(sr_ensemble_best_models, label="SR Ensemble Best Models")
    plt.xlabel("Traces")
    plt.ylabel("Success Rate (%)")
    plt.xlim([0, kr_nt])
    plt.legend()
    plt.show()
