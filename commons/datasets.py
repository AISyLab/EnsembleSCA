class SCADatasets:

    def __init__(self):
        self.trace_set_list = []

    def get_trace_set(self, trace_set_name):
        trace_list = self.get_trace_set_list()
        return trace_list[trace_set_name]

    def get_trace_set_list(self):
        parameters_ascad_fixed_key = {
            "file": "ASCAD.h5",
            "key": "4DFBE0F27221FE10A78D4ADC8E490469",
            "key_offset": 32,
            "input_offset": 0,
            "data_length": 32,
            "first_sample": 0,
            "number_of_samples": 700,
            "n_profiling": 50000,
            "n_attack": 10000,
            "classes": 9,
            "good_key": 224,
            "number_of_key_hypothesis": 256,
            "epochs": 50,
            "mini-batch": 50
        }

        parameters_ascad_random_key = {
            "file": "ascad-variable.h5",
            "key": "00112233445566778899AABBCCDDEEFF",
            "key_offset": 16,
            "input_offset": 0,
            "data_length": 50,
            "first_sample": 0,
            "number_of_samples": 1400,
            "n_profiling": 100000,
            "n_attack": 1000,
            "classes": 9,
            "good_key": 34,
            "number_of_key_hypothesis": 256,
            "epochs": 50,
            "mini-batch": 400
        }

        parameters_ches_ctf = {
            "file": "ches_ctf.h5",
            "key": "2EEE5E799D72591C4F4C10D8287F397A",
            "key_offset": 32,
            "input_offset": 0,
            "data_length": 48,
            "first_sample": 0,
            "number_of_samples": 2200,
            "n_profiling": 45000,
            "n_attack": 5000,
            "classes": 9,
            "good_key": 46,
            "number_of_key_hypothesis": 256,
            "epochs": 50,
            "mini-batch": 400
        }

        self.trace_set_list = {
            "ascad_fixed_key": parameters_ascad_fixed_key,
            "ascad_random_key": parameters_ascad_random_key,
            "ches_ctf": parameters_ches_ctf
        }

        return self.trace_set_list
