import os
import torch
from torch.utils.data import Dataset
import librosa


signal_train_clean_folder = './data/signal_train_clean_folder'
signal_train_noisy_folder = './data/signal_train_noisy_folder'

signal_test_clean_folder = './data/signal_test_clean_folder'
signal_test_noisy_folder = './data/signal_test_noisy_folder'


class AudioDataset(Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):
        if data_type == 'train':
            clean_path = signal_train_clean_folder
            noisy_path = signal_train_noisy_folder
        elif data_type == 'test':
            clean_path = signal_test_clean_folder
            noisy_path = signal_test_noisy_folder
        else:
            raise ValueError

        if not os.path.exists(clean_path) or not os.path.exists(noisy_path):
            raise FileNotFoundError('The {} data folder does not exist!'.format(data_type))

        self.data_type = data_type
        self.name_list = os.listdir(clean_path)
        self.clean_file_names = [os.path.join(clean_path, filename) for filename in self.name_list]
        self.noisy_file_names = [os.path.join(noisy_path, filename) for filename in self.name_list]

    def __getitem__(self, idx):
        clean_y, _ = librosa.load(self.clean_file_names[idx], sr=16000)
        noisy_y, _ = librosa.load(self.noisy_file_names[idx], sr=16000)
        clean_t = torch.from_numpy(clean_y)
        noisy_t = torch.from_numpy(noisy_y)
        if self.data_type == 'train':
            return clean_t, noisy_t
        else:
            return os.path.basename(self.name_list[idx]), clean_t, noisy_t

    def __len__(self):
        return len(self.name_list)

