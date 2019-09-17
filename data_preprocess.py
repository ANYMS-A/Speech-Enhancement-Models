import os
import librosa
import numpy as np
from tqdm import tqdm
from utils import signal_to_spectrogram, log_power_spectrogram, get_phase, emphasis

clean_train_folder = './data/clean_trainset_28spk_wav'
noisy_train_folder = './data/noisy_trainset_28spk_wav'
clean_test_folder = './data/clean_testset_wav'
noisy_test_folder = './data/noisy_testset_wav'

signal_train_folder = './data/signal_train_data'
signal_test_folder = './data/signal_test_data'
spec_train_folder = './data/spec_train_data'
spec_test_folder = './data/spec_test_data'

window_size = 2 ** 14  # about 1 second of samples
sample_rate = 16000

folder_list = [signal_train_folder, signal_test_folder, spec_train_folder, spec_test_folder]
for f in folder_list:
    if not os.path.exists(f):
        os.makedirs(f)


def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, sr = librosa.load(file, sr=sample_rate)
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices


def process_and_serialize(data_type):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5

    if data_type == 'train':
        clean_folder = clean_train_folder
        noisy_folder = noisy_train_folder
        signal_save_folder = signal_train_folder
        spec_save_folder = spec_train_folder
    else:
        clean_folder = clean_test_folder
        noisy_folder = noisy_test_folder
        signal_save_folder = signal_test_folder
        spec_save_folder = spec_test_folder

    # walk through the path, slice the audio file, and save the serialized result
    for root, dirs, files in os.walk(clean_folder):
        if len(files) == 0:
            continue
        for filename in tqdm(files, desc='Serialize and down-sample {} audios'.format(data_type)):
            clean_file = os.path.join(clean_folder, filename)
            noisy_file = os.path.join(noisy_folder, filename)
            # slice both clean signal and noisy signal
            clean_sliced = slice_signal(clean_file, window_size, stride, sample_rate)
            noisy_sliced = slice_signal(noisy_file, window_size, stride, sample_rate)
            # serialize - file format goes [original_file]_[slice_number].npy
            # ex) p293_154.wav_5.npy denotes 5th slice of p293_154.wav file
            for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
                clean_empha, noisy_empha = emphasis(slice_tuple[0], pre=True), emphasis(slice_tuple[1], pre=True)
                # save the signal pair
                signal_pair = np.array([clean_empha, noisy_empha])
                np.save(os.path.join(signal_save_folder, '{}_{}'.format(filename, idx)), arr=signal_pair)
                # save the spectrogram pair
                clean_spec = log_power_spectrogram(signal_to_spectrogram(clean_empha))
                noisy_spec = log_power_spectrogram(signal_to_spectrogram(noisy_empha))
                noisy_phase = get_phase(signal_to_spectrogram(slice_tuple[1]))
                spec_pair = np.array([clean_spec, noisy_spec, noisy_phase])
                np.save(os.path.join(spec_save_folder, '{}_{}'.format(filename, idx)), arr=spec_pair)


def data_verify(data_type):
    """
    Verifies the length of each data after pre-process.
    """
    if data_type == 'train':
        serialized_folder = signal_train_folder
    else:
        serialized_folder = signal_test_folder

    for root, dirs, files in os.walk(serialized_folder):
        for filename in tqdm(files, desc='Verify serialized {} audios'.format(data_type)):
            data_pair = np.load(os.path.join(root, filename))
            if data_pair.shape[1] != window_size:
                print('Snippet length not {} : {} instead'.format(window_size, data_pair.shape[1]))
                break


if __name__ == '__main__':
    process_and_serialize('train')
    data_verify('train')
    process_and_serialize('test')
    data_verify('test')

