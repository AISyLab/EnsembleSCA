from tensorflow.keras import backend as backend
from tensorflow.keras.utils import to_categorical
from commons.neural_networks import NeuralNetwork
from commons.sca_metrics import SCAMetrics
from commons.datasets import SCADatasets
from commons.load_datasets import LoadDatasets
import numpy as np
import random


class EnsembleAES:

    def __init__(self):
        self.number_of_models = 50
        self.number_of_best_models = 10
        self.ge_all_validation = []
        self.ge_all_attack = []
        self.sr_all_validation = []
        self.sr_all_attack = []
        self.k_ps_all = []
        self.ge_ensemble = None
        self.ge_ensemble_best_models = None
        self.ge_best_model_validation = None
        self.ge_best_model_attack = None
        self.sr_ensemble = None
        self.sr_ensemble_best_models = None
        self.sr_best_model_validation = None
        self.sr_best_model_attack = None
        self.target_dataset = None
        self.l_model = None
        self.target_byte = None
        self.classes = None
        self.epochs = None
        self.mini_batch = None

    def set_dataset(self, target):
        self.target_dataset = target

    def set_leakage_model(self, leakage_model):
        self.l_model = leakage_model
        if leakage_model == "HW":
            self.classes = 9
        else:
            self.classes = 256

    def set_target_byte(self, target_byte):
        self.target_byte = target_byte

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_mini_batch(self, mini_batch):
        self.mini_batch = mini_batch

    def __add_if_one(self, value):
        return 1 if value == 1 else 0

    def get_best_models(self, n_models, result_models_validation, n_traces):
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

    def run_mlp(self, X_profiling, Y_profiling, X_validation, Y_validation, X_attack, Y_attack, plt_validation, plt_attack, params,
                step, fraction):
        mini_batch = random.randrange(500, 1000, 100)
        learning_rate = random.uniform(0.0001, 0.001)
        activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
        layers = random.randrange(2, 8, 1)
        neurons = random.randrange(500, 800, 100)

        model = NeuralNetwork().mlp_random(self.classes, params["number_of_samples"], activation, neurons, layers, learning_rate)
        model.fit(
            x=X_profiling,
            y=Y_profiling,
            batch_size=self.mini_batch,
            verbose=1,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(X_validation, Y_validation),
            callbacks=[])

        ge_validation, sr_validation, kp_krs = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte,
                                                                      X_validation, plt_validation, step, fraction)
        ge_attack, sr_attack, _ = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte, X_attack, plt_attack, step,
                                                         fraction)

        backend.clear_session()

        return ge_validation, ge_attack, sr_validation, sr_attack, kp_krs

    def run_cnn(self, X_profiling, Y_profiling, X_validation, Y_validation, X_attack, Y_attack, plt_validation, plt_attack, params,
                step, fraction):
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))

        mini_batch = random.randrange(500, 1000, 100)
        learning_rate = random.uniform(0.0001, 0.001)
        activation = ['relu', 'tanh', 'elu', 'selu'][random.randint(0, 3)]
        dense_layers = random.randrange(2, 8, 1)
        neurons = random.randrange(500, 800, 100)
        conv_layers = random.randrange(1, 2, 1)
        filters = random.randrange(8, 32, 4)
        kernel_size = random.randrange(10, 20, 2)
        stride = random.randrange(5, 10, 5)

        model = NeuralNetwork().cnn_random(self.classes, params["number_of_samples"], activation, neurons, conv_layers, filters,
                                           kernel_size, stride, dense_layers, learning_rate)
        model.fit(
            x=X_profiling,
            y=Y_profiling,
            batch_size=self.mini_batch,
            verbose=1,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(X_validation, Y_validation),
            callbacks=[])

        ge_validation, sr_validation, kp_krs = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte,
                                                                      X_validation, plt_validation,
                                                                      step, fraction)
        ge_attack, sr_attack, _ = SCAMetrics().ge_and_sr(100, model, params, self.l_model, self.target_byte, X_attack, plt_attack, step,
                                                         fraction)

        backend.clear_session()

        return ge_validation, ge_attack, sr_validation, sr_attack, kp_krs

    def compute_ensembles(self, kr_nt, correct_key):

        list_of_best_models = self.get_best_models(self.number_of_models, self.ge_all_validation, kr_nt)

        self.ge_best_model_validation = self.ge_all_validation[list_of_best_models[0]]
        self.ge_best_model_attack = self.ge_all_attack[list_of_best_models[0]]
        self.sr_best_model_validation = self.sr_all_validation[list_of_best_models[0]]
        self.sr_best_model_attack = self.sr_all_attack[list_of_best_models[0]]

        kr_ensemble = np.zeros(kr_nt)
        krs_ensemble = np.zeros((100, kr_nt))
        kr_ensemble_best_models = np.zeros(kr_nt)
        krs_ensemble_best_models = np.zeros((100, kr_nt))

        for run in range(100):

            key_p_ensemble = np.zeros(256)
            key_p_ensemble_best_models = np.zeros(256)

            for index in range(kr_nt):
                for model_index in range(self.number_of_models):
                    key_p_ensemble += np.log(self.k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)
                for model_index in range(self.number_of_best_models):
                    key_p_ensemble_best_models += np.log(self.k_ps_all[list_of_best_models[model_index]][run][index] + 1e-36)

                key_p_ensemble_sorted = np.argsort(key_p_ensemble)[::-1]
                key_p_ensemble_best_models_sorted = np.argsort(key_p_ensemble_best_models)[::-1]

                kr_position = list(key_p_ensemble_sorted).index(correct_key) + 1
                kr_ensemble[index] += kr_position
                krs_ensemble[run][index] = kr_position

                kr_position = list(key_p_ensemble_best_models_sorted).index(correct_key) + 1
                kr_ensemble_best_models[index] += kr_position
                krs_ensemble_best_models[run][index] = kr_position

            print("Run {} - GE {} models: {} | GE {} models: {} | ".format(run, self.number_of_models,
                                                                           int(kr_ensemble[kr_nt - 1] / (run + 1)),
                                                                           self.number_of_best_models,
                                                                           int(kr_ensemble_best_models[kr_nt - 1] / (run + 1))))

        ge_ensemble = kr_ensemble / 100
        ge_ensemble_best_models = kr_ensemble_best_models / 100

        sr_ensemble = np.zeros(kr_nt)
        sr_ensemble_best_models = np.zeros(kr_nt)

        for index in range(kr_nt):
            for run in range(100):
                sr_ensemble[index] += self.__add_if_one(krs_ensemble[run][index])
                sr_ensemble_best_models[index] += self.__add_if_one(krs_ensemble_best_models[run][index])

        return ge_ensemble, ge_ensemble_best_models, sr_ensemble/100, sr_ensemble_best_models/100

    def create_z_score_norm(self, dataset):
        z_score_mean = np.mean(dataset, axis=0)
        z_score_std = np.std(dataset, axis=0)
        return z_score_mean, z_score_std

    def apply_z_score_norm(self, dataset, z_score_mean, z_score_std):
        for index in range(len(dataset)):
            dataset[index] = (dataset[index] - z_score_mean) / z_score_std

    def run_ensemble(self, number_of_models, number_of_best_models):

        self.number_of_models = number_of_models
        self.number_of_best_models = number_of_best_models

        target_params = SCADatasets().get_trace_set(self.target_dataset)

        root_folder = "D:/traces/"

        (X_profiling, Y_profiling), (X_validation, Y_validation), (X_attack, Y_attack), (
            _, plt_validation, plt_attack) = LoadDatasets().load_dataset(
            root_folder + target_params["file"], target_params["n_profiling"], target_params["n_attack"], self.target_byte, self.l_model)

        # normalize with z-score
        z_score_mean, z_score_std = self.create_z_score_norm(X_profiling)
        self.apply_z_score_norm(X_profiling, z_score_mean, z_score_std)
        self.apply_z_score_norm(X_validation, z_score_mean, z_score_std)
        self.apply_z_score_norm(X_attack, z_score_mean, z_score_std)

        # convert labels to categorical labels
        Y_profiling = to_categorical(Y_profiling, num_classes=self.classes)
        Y_validation = to_categorical(Y_validation, num_classes=self.classes)
        Y_attack = to_categorical(Y_attack, num_classes=self.classes)

        X_profiling = X_profiling.astype('float32')
        X_validation = X_validation.astype('float32')
        X_attack = X_attack.astype('float32')

        kr_step = 10  # key rank processed for each kr_step traces
        kr_fraction = 1  # validation or attack sets are divided by kr_fraction before computing key rank

        self.ge_all_validation = []
        self.sr_all_validation = []
        self.ge_all_attack = []
        self.k_ps_all = []

        kr_nt = int(len(X_validation) / (kr_step * kr_fraction))

        # train random MLP
        for model_index in range(self.number_of_models):
            ge_validation, ge_attack, sr_validation, sr_attack, kp_krs = self.run_mlp(X_profiling, Y_profiling,
                                                                                      X_validation, Y_validation,
                                                                                      X_attack, Y_attack,
                                                                                      plt_validation, plt_attack,
                                                                                      target_params, kr_step, kr_fraction)
            self.ge_all_validation.append(ge_validation)
            self.ge_all_attack.append(ge_attack)
            self.sr_all_validation.append(sr_validation)
            self.sr_all_attack.append(sr_attack)
            self.k_ps_all.append(kp_krs)

        # train random CNN
        # for model_index in range(self.number_of_models):
        #     ge_validation, ge_attack, sr_validation, sr_attack, kp_krs = self.run_cnn(X_profiling, Y_profiling,
        #                                                                               X_validation, Y_validation,
        #                                                                               X_attack, Y_attack,
        #                                                                               plt_validation, plt_attack,
        #                                                                               target_params, kr_step, kr_fraction)
        #     self.ge_all_validation.append(ge_validation)
        #     self.ge_all_attack.append(ge_attack)
        #     self.sr_all_validation.append(sr_validation)
        #     self.sr_all_attack.append(sr_attack)
        #     self.k_ps_all.append(kp_krs)

        ge_ensemble, ge_ensemble_best_models, sr_ensemble, sr_ensemble_best_models = self.compute_ensembles(kr_nt,
                                                                                                            target_params["good_key"])

        self.ge_ensemble = ge_ensemble
        self.ge_ensemble_best_models = ge_ensemble_best_models
        self.sr_ensemble = sr_ensemble
        self.sr_ensemble_best_models = sr_ensemble_best_models

    def get_ge_ensemble(self):
        return self.ge_ensemble

    def get_ge_ensemble_best_models(self):
        return self.ge_ensemble_best_models

    def get_ge_best_model_validation(self):
        return self.ge_best_model_validation

    def get_ge_best_model_attack(self):
        return self.ge_best_model_attack

    def get_sr_ensemble(self):
        return self.sr_ensemble

    def get_sr_ensemble_best_models(self):
        return self.sr_ensemble_best_models

    def get_sr_best_model_validation(self):
        return self.sr_best_model_validation

    def get_sr_best_model_attack(self):
        return self.sr_best_model_attack
