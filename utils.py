import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import torch.nn as nn


def draw_loss(tl, vl, epoch, model_name):

    plt.figure(figsize=(12, 6))
    x1 = np.linspace(0, epoch, len(tl))
    plt.plot(x1, tl, 'r--', label=f'train_loss of {model_name}')
    x2 = np.linspace(0, epoch, len(vl))
    plt.plot(x2, vl, 'g--', label=f'valid_loss of {model_name}')
    plt.title('Train and Validation Loss of ' + model_name)
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../figures/{model_name}_loss.png")
    plt.close()
    return


def draw_gan_loss(dcl, dnl, gl, l1, epoch, using_l1, model_name, loss_type):
    if isinstance(loss_type, nn.MSELoss):
        description = 'MSE'
    elif isinstance(loss_type, nn.BCEWithLogitsLoss):
        description = 'BCE'
    else:
        raise TypeError('No such loss type!')

    # G_train_loss,  D_train_loss
    plt.figure(figsize=(12, 6))

    # draw D_noisy loss
    x2 = np.linspace(6, epoch, len(dnl))
    plt.plot(x2, dnl, 'g--', label=f'D Loss of enhanced speech')
    # draw g_loss_
    x3 = np.linspace(9, epoch, len(gl))
    plt.plot(x3, gl, 'b--', label='G Loss')
    # draw L1 loss
    x4 = np.linspace(0, epoch, len(l1))
    plt.plot(x4, l1, 'k--', label='L1 distance between clean and enhanced spectrogram')
    if using_l1:

        plt.title(f'Loss of {model_name} with L1 restriction')
    else:
        plt.title(f'Loss of {model_name} without L1 restriction')
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    if using_l1:
        plt.savefig(f"../figures/{model_name}_loss_L1.png")
    else:
        plt.savefig(f'../figures/{model_name}_loss_No_L1.png')
    plt.close()
    return


def emphasis(signal, emph_coeff=0.95, pre=True):
    """
    Pre-emphasis or De-emphasis of higher frequencies given a batch of signal.

    Args:
        signal: signals, represented as numpy arrays
        emph_coeff: emphasis coefficient
        pre: pre-emphasis or de-emphasis signals

    Returns:
        result: pre-emphasized or de-emphasized signal
    """
    if pre:
        x0 = np.reshape(signal[0], (1,))
        diff = signal[1:] - emph_coeff * signal[:-1]
        concat = np.concatenate((x0, diff), axis=0)
        return concat
    else:
        x = np.zeros(signal.shape[0], dtype=np.float32)
        x[0] = signal[0]
        for n in range(1, signal.shape[0], 1):
            x[n] = emph_coeff * x[n - 1] + signal[n]
        return x


def recover_signal(file_list, window_size, stride, sample_rate):
    """

    Helper function for recover the sliced enhanced signal
    """
    pass


def signal_to_spectrogram(signal):
    spec = librosa.stft(signal, n_fft=512, win_length=32, hop_length=16, window='hann')
    return spec


# complex to log-power-spectrogram
def log_power_spectrogram(spec):
    lps = np.log(np.power(np.abs(spec), 2))
    return lps


def get_phase(complex_spectrogram):
    phase = np.angle(complex_spectrogram)
    return phase


# log-power-spectrogram to magnitude
def lps_to_mag(lps):
    mag = np.power(np.exp(lps), 1 / 2)
    mag = mag - 1e-20
    return mag


# recover complex spectrogram from magnitude
def magnitude_to_complex(magnitude_spectrogram, phase):
    complex_spectrogram = magnitude_spectrogram * np.exp(1j * phase)
    return complex_spectrogram


# complex to magnitude
def complex_to_mag(spec):
    magnitude = np.abs(spec)
    return magnitude


# helper function test the in & out of model
def test_in_n_out(model, in_size, z=None):
    t = torch.randn(128, 2, in_size)
    if z is None:
        out = model(t)
    else:
        out = model(t, z)
    print(out.size())
    return


