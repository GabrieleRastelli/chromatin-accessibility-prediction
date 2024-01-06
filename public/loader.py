import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import hickle as hkl


def one_hot(seq):
    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]}

    code = np.empty(shape=(len(seq), 4), dtype=np.int8)
    for location, base in enumerate(seq, start=0):
        code[location] = base_map[base]

    return code


class Seq2Ab(Dataset):
    def __init__(self):

        super(Seq2Ab, self).__init__()
        self.inputFile = None
        self.labelFile = None
        self.sequence = None
        self.labels = None
        self.sequences = None
        self.label = None
        self.label_value = "float"

    def load_data_hkl(self, input_files):
        self.data = {}
        ProgressBar = tqdm(input_files)
        for input_file in ProgressBar:
            ProgressBar.set_description("Loading file %s" % input_file)
            loaded_data = hkl.load(input_file)

            # Iterate through keys in the loaded_data dictionary
            for key in loaded_data.keys():
                if key in self.data:
                    # Append along the common dimension (axis=0)
                    self.data[key] = np.concatenate((self.data[key], loaded_data[key]), axis=0)
                else:
                    # If the key is not present, add it to self.data
                    self.data[key] = loaded_data[key]
        print('Hkl loading done!')

    def set_input_and_target(self, indices):
        self.data = {key: value[indices] for key, value in self.data.items()}
        x = self.data['mat']
        x_kspec = self.data['kmer']

        x_kspec = x_kspec.reshape((x_kspec.shape[0], 1024, 4))
        x = np.concatenate((x, x_kspec), axis=1)
        x = x[:, np.newaxis]
        self.x = x.transpose((0, 1, 3, 2))
        self.labels = self.data['y']
        print('Data loading done!')

    def __getitem__(self, item):
        cur_x = self.x[item]
        cur_label = self.labels[item]
        return cur_x, cur_label

    def __len__(self):
        return len(self.x)


class SampleLoader:
    def __init__(self):
        self.mode = None
        self.loader = None

    def __call__(self):
        return self.loader
