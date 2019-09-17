import os
import librosa
from tqdm import tqdm
from utils import emphasis


clean_train_folder = '../SEGAN_github/data/clean_trainset_28spk_wav'
noisy_train_folder = '../SEGAN_github/data/noisy_trainset_28spk_wav'
clean_test_folder = '../SEGAN_github/data/clean_testset_wav'
noisy_test_folder = '../SEGAN_github/data/noisy_testset_wav'

signal_train_clean_folder = './data/signal_train_clean_folder'
signal_train_noisy_folder = './data/signal_train_noisy_folder'

signal_test_clean_folder = './data/signal_test_clean_folder'
signal_test_noisy_folder = './data/signal_test_noisy_folder'

window_size = 2 ** 14  # about 1 second of samples
sample_rate = 16000

folder_list = [signal_train_clean_folder, signal_train_noisy_folder, signal_test_clean_folder, signal_test_noisy_folder]
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
        clean_save_folder = signal_train_clean_folder
        noisy_save_folder = signal_train_noisy_folder

    else:
        clean_folder = clean_test_folder
        noisy_folder = noisy_test_folder
        clean_save_folder = signal_test_clean_folder
        noisy_save_folder = signal_test_noisy_folder

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
                clean_empha = emphasis(slice_tuple[0], pre=True)
                noisy_empha = emphasis(slice_tuple[1], pre=True)
                # save the signal
                librosa.output.write_wav(os.path.join(clean_save_folder, '{}_{}.wav'.format(filename.replace('.wav', ''), idx)), clean_empha, sr=16000)
                librosa.output.write_wav(os.path.join(noisy_save_folder, '{}_{}.wav'.format(filename.replace('.wav', ''), idx)), noisy_empha, sr=16000)


if __name__ == '__main__':
    process_and_serialize('train')
    process_and_serialize('test')


