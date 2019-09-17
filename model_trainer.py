from resgan_model import ResDiscriminator, ResGenerator, ResGenerator2D, ResDiscriminator2D
from mlp_model import SEMLP
import torch
from datasets import AudioDataset
from torch.utils.data import random_split, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import draw_gan_loss, draw_loss, emphasis, lps_to_mag, magnitude_to_complex, signal_to_spectrogram, get_phase
import numpy as np
import os
from scipy.io import wavfile
import librosa
from simple_generator import SimpleGenerator


class BaseTrainer(object):
    def __init__(self, **kwargs):
        self.model_name = kwargs['model_name']
        print(self.model_name)
        self.num_epochs = kwargs['num_epochs']

        self.num_GPU = kwargs['num_GPU']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.pin_memory = kwargs['pin_memory']
        self.lr = kwargs['lr']
        self.using_spectrogram = kwargs['using_spectrogram']
        self.start_GPU = kwargs['start_GPU']
        self.device = torch.device(f"cuda:{self.start_GPU}" if (torch.cuda.is_available() and self.num_GPU > 0) else "cpu")

        optimizers = {'Adam': optim.Adam, 'SGD': optim.SGD}
        self.optimizer_name = kwargs['optimizer']
        self.optimizer = optimizers[kwargs['optimizer']]
        self.betas = (0.5, 0.999)
        criterions = {'BCE': nn.BCEWithLogitsLoss(), 'MSE': nn.MSELoss()}
        self.criterion_name = kwargs['criterion']
        self.criterion = criterions[kwargs['criterion']]
        self.using_l1 = kwargs['using_l1']  # bool'
        self.using_simple_g = kwargs['using_simple_g']

        # early stopping config
        self.early_stopping_patient = 0
        self.converge_threshold = kwargs['converge_threshold']
        # init data set
        audio_dataset = AudioDataset(data_type='train')
        # train & valid split
        total_len = len(audio_dataset)
        train_len = int(total_len * 0.8)
        valid_len = total_len - train_len
        # split train & valid datasets
        self.train_dataset, self.valid_dataset = random_split(audio_dataset, [train_len, valid_len])
        self.test_dataset = AudioDataset(data_type='test')

        self.train_data_loader = DataLoader(dataset=self.train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory)

        self.valid_data_loader = DataLoader(dataset=self.valid_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory)
        # set test_data batch_size 1 for convenient
        self.test_data_loader = DataLoader(dataset=self.test_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory)


class ModelTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ModelTrainer, self).__init__(**kwargs)

    def _train_gan_model(self):
        print(f'Start training {self.model_name}')

        if self.using_spectrogram:
            if 'GAN' in self.model_name:
                # create D and G instances
                discriminator = ResDiscriminator2D().to(self.device)
                generator = ResGenerator2D().to(self.device)

            elif 'MLP' in self.model_name:
                discriminator = ResDiscriminator2D().to(self.device)
                generator = SEMLP(in_size=1799, out_size=257, hidden_size=1024, num_layer=3).to(self.device)
                print('MLP created as generator')
        else:
            if not self.using_simple_g:
                generator = ResGenerator().to(self.device)
            elif self.using_simple_g:
                generator = SimpleGenerator().to(self.device)
                print('using simple generator')
            else:
                raise ValueError

            discriminator = ResDiscriminator().to(self.device)

        if (self.device.type == 'cuda') and (self.num_GPU > 1):
            discriminator = nn.DataParallel(discriminator, list(range(self.start_GPU, self.start_GPU + self.num_GPU)))
            generator = nn.DataParallel(generator, list(range(self.start_GPU, self.start_GPU + self.num_GPU)))

        print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
        print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))
        if self.optimizer_name == 'Adam':
            g_optimizer = self.optimizer(generator.parameters(), lr=self.lr, betas=self.betas)
            d_optimizer = self.optimizer(discriminator.parameters(), lr=self.lr * 0.9, betas=self.betas)
        elif self.optimizer_name == 'SGD':
            g_optimizer = self.optimizer(generator.parameters(), lr=self.lr)
            d_optimizer = self.optimizer(discriminator.parameters(), lr=self.lr * 0.9)

        criterion_l1 = nn.L1Loss()
        lambda_constant = 100
        real_label = 1
        fake_label = 0

        print('Start training!')
        G_losses = []
        L1_losses = []
        D_losses_clean = []
        D_losses_noisy = []
        D_x_history = []
        D_G_z1_history = []
        D_G_z2_history = []
        G_valid_losses = []

        for epoch in range(self.num_epochs):
            train_bar = tqdm(self.train_data_loader)
            generator.train()
            discriminator.train()
            for iteration, train_data in enumerate(train_bar):
                train_data = self._prepare_train_data(train_data)
                clean_data = train_data[0]
                noisy_data = train_data[1]
                if 'MLP' in self.model_name:
                    clean_data = clean_data.transpose(dim0=1, dim1=2).unsqueeze(
                        dim=1).detach()  # B x 1025 x 257 -> B x 1 x 257 x 1025
                    noisy_data = noisy_data.transpose(dim0=1, dim1=2).unsqueeze(
                        dim=1) .detach()  # B x 1025 x 257 -> B x 1 x 257 x 1025
                    noisy_feature_data = train_data[2].unsqueeze(dim=1).detach()  # B x 1 x 1025 x 1799
                # TRAIN D to recognize clean audio as clean

                discriminator.zero_grad()
                # discriminator forward with clean speech
                outputs = discriminator(torch.cat((clean_data, noisy_data), 1)).view(-1)
                # label for calculating loss
                label = torch.full((outputs.size()[0],), real_label, device=self.device)
                clean_loss = self.criterion(outputs, label)  # minimize it with label=1
                clean_loss.backward()
                # append to history
                D_losses_clean.append(clean_loss.item())
                # the probability description
                D_x = outputs.detach().mean().item()
                D_x_history.append(D_x)

                # TRAIN D to recognize generated audio as noisy
                if 'MLP' in self.model_name:
                    generated_outputs = generator(noisy_feature_data).transpose(dim0=2, dim1=3)
                else:
                    generated_outputs = generator(noisy_data)
                # discriminator forward with fake clean speech
                # you must detach generated output out of this backward pass
                outputs = discriminator(torch.cat((generated_outputs.detach(), noisy_data), 1)).view(-1)
                label.fill_(fake_label)
                noisy_loss = self.criterion(outputs, label)  # minimize it with label=0
                noisy_loss.backward()
                # append to history
                D_losses_noisy.append(noisy_loss.item())
                # probability description
                D_G_z1 = outputs.detach().mean().item()
                D_G_z1_history.append(D_G_z1)
                # D has accumulate the gradient of clean and noisy loss in the .grad
                d_optimizer.step()  # update parameters


                # TRAIN G so that D recognizes G(z) as real
                generator.zero_grad()
                if 'MLP' in self.model_name:
                    generated_outputs = generator(noisy_feature_data).transpose(dim0=2, dim1=3)
                else:
                    generated_outputs = generator(noisy_data)
                outputs = discriminator(torch.cat((generated_outputs, noisy_data), 1)).view(-1)
                # using reverse-label trick!
                label.fill_(real_label)
                g_loss = self.criterion(outputs, label)
                # L1 loss of Generator
                l1_loss = criterion_l1(generated_outputs, clean_data)
                if self.using_l1:
                    total_loss = g_loss + lambda_constant * l1_loss
                else:
                    total_loss = g_loss
                # back-propagation + optimize
                total_loss.backward()
                g_optimizer.step()
                # append to G loss history
                L1_losses.append(l1_loss.item())
                G_losses.append(g_loss.item())
                # probability description
                D_G_z2 = outputs.detach().mean().item()
                D_G_z2_history.append(D_G_z2)
                # tqdm process bar
                train_bar.set_description(
                    'Epoch {}: d_clean_loss:{:.4f},d_noisy_loss{:.4f}, g_loss:{:.4f}, l1_loss{:.4f}, D_x:{:.4f}, D_G_z1:{:.4f}, D_G_z2:{:.4f}'
                    .format(epoch + 1,
                            clean_loss.item(),
                            noisy_loss.item(),
                            g_loss.item(),
                            l1_loss.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2))

            # draw training loss
            draw_gan_loss(D_losses_clean[::50], D_losses_noisy[::50], G_losses[::50], L1_losses[::50],
                          epoch=int(epoch), using_l1=self.using_l1, model_name=self.model_name, loss_type=self.criterion)
            # validation
            mean_valid_loss = self._valid_gan_model(generator=generator, epoch=epoch)
            G_valid_losses.append(mean_valid_loss)
            draw_loss(G_losses, G_valid_losses, epoch=int(epoch), model_name=self.model_name)

            # converge checking
            # self._converge_checking(G_losses)
            # # early stopping
            # self._early_stopping(G_valid_losses)
            #
            # if self.early_stopping_patient >= 2:
            #     print('Early stopping training, discard the rest epochs')
            #     break

        if epoch == self.num_epochs - 1:
            print('All epochs trained, no early stopping')
        self._test_and_save(generator, epoch=epoch)
        print(f'Training {self.model_name} Finished!')

    def _only_train_g(self, generator, clean_data, noisy_data, g_optimizer):

        if 'MLP' in self.model_name:
            generated_outputs = generator(noisy_data).transpose(dim0=2, dim1=3)
        else:
            generated_outputs = generator(noisy_data)

        total_loss = F.mse_loss(generated_outputs, clean_data)
        l1_loss = F.l1_loss(generated_outputs, clean_data)
        total_loss.backward()
        g_optimizer.step()
        generator.zero_grad()
        return l1_loss

    def _train_autoencoder(self):
        if '1D' in self.model_name:
            model = ResGenerator()
        elif '2D' in self.model_name:
            model = ResGenerator2D()
        elif 'simple' in self.model_name:
            model = SimpleGenerator()

        model = model.to(self.device)
        if (self.device.type == 'cuda') and (self.num_GPU > 1):
            model = nn.DataParallel(model, list(range(self.start_GPU, self.start_GPU + self.num_GPU)))

        print(f'start training {self.model_name}')
        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        criterion = self.criterion
        train_loss_history = []
        valid_loss_history = []
        for epoch in range(self.num_epochs):
            train_bar = tqdm(self.train_data_loader)
            for train_data in train_bar:
                train_data = self._prepare_train_data(train_data)
                clean_data = train_data[0].detach()
                noisy_data = train_data[1].detach()
                optimizer.zero_grad()
                # model forward propagation
                outputs = model(noisy_data)  # spec or waveform
                # loss = criterion(outputs, clean_data)
                loss = F.l1_loss(outputs, clean_data)
                loss.backward()
                optimizer.step()
                train_loss_history.append(loss.detach().item())
                train_bar.set_description(f'Epoch:{epoch}, training_loss:{loss.detach().item()}')

            # validation
            one_epoch_valid = self._valid_autoencoder_model(model, epoch=epoch)
            valid_loss_history = valid_loss_history + one_epoch_valid
            # draw loss
            draw_loss(train_loss_history, valid_loss_history, epoch=epoch, model_name=self.model_name)

        self._test_and_save(model=model, epoch=epoch)

    def _valid_autoencoder_model(self, model, epoch):
        with torch.no_grad():
            model.eval()
            valid_loss_list = []
            valid_bar = tqdm(self.valid_data_loader)
            for valid_data in valid_bar:
                valid_data = self._prepare_train_data(valid_data)
                clean_data = valid_data[0].detach()  # B x 1025 x 257
                noisy_data = valid_data[1].detach()  # B x 1025 x 1799
                outputs = model(noisy_data)
                loss = F.mse_loss(outputs, clean_data)
                valid_loss_list.append(loss.item())
                valid_bar.set_description(f'Epoch:{epoch}, valid_loss:{loss.detach().item()}')

        return valid_loss_list

    def _train_mlp_model(self):
        model = SEMLP(in_size=1799,
                      out_size=257,
                      hidden_size=1024,
                      num_layer=3).to(self.device)
        if (self.device.type == 'cuda') and (self.num_GPU > 1):
            model = nn.DataParallel(model, list(range(self.start_GPU, self.start_GPU + self.num_GPU)))

        optimizer = self.optimizer(model.parameters(), lr=self.lr)
        criterion = self.criterion
        train_loss_history = []
        valid_loss_history = []
        for epoch in range(self.num_epochs):
            train_bar = tqdm(self.train_data_loader)
            for train_data in train_bar:
                train_data = self._prepare_train_data(train_data)
                clean_data = train_data[0].detach()  # B x 1025 x 257
                noisy_data = train_data[1].detach()  # B x 1025 x 1799
                optimizer.zero_grad()
                # model forward propagation
                outputs = model(noisy_data)  # outputs: B x 1025 x 257
                # loss = criterion(outputs, clean_data)
                loss = F.l1_loss(outputs, clean_data)
                loss.backward()
                optimizer.step()
                train_loss_history.append(loss.detach().item())
                train_bar.set_description(f'Epoch:{epoch}, training_loss:{loss.detach().item()}')

            # validation
            one_epoch_valid = self._valid_model(model, epoch=epoch)
            valid_loss_history = valid_loss_history + one_epoch_valid

            # draw loss
            draw_loss(train_loss_history, valid_loss_history, epoch=epoch, model_name=self.model_name)

        self._test_and_save(model=model, epoch=epoch)

    def _valid_model(self, model, epoch):
        with torch.no_grad():
            model.eval()
            valid_loss_list = []
            valid_bar = tqdm(self.valid_data_loader)
            for valid_data in valid_bar:
                valid_data = self._prepare_train_data(valid_data)
                clean_data = valid_data[0].detach()  # B x 1025 x 257
                noisy_data = valid_data[1].detach()  # B x 1025 x 1799
                outputs = model(noisy_data)

                loss = F.mse_loss(outputs, clean_data)

                valid_loss_list.append(loss.item())
                valid_bar.set_description(f'Epoch:{epoch}, valid_loss:{loss.detach().item()}')

        return valid_loss_list

    def _valid_gan_model(self, generator, epoch):
        with torch.no_grad():
            generator.eval()
            valid_loss_history = []
            valid_bar = tqdm(self.valid_data_loader, desc='valid GAN model and save the validation loss')
            for valid_data in valid_bar:
                valid_data = self._prepare_train_data(valid_data)
                clean_data = valid_data[0]
                noisy_data = valid_data[1]
                if 'MLP' in self.model_name:
                    clean_data = clean_data.transpose(dim0=1, dim1=2).unsqueeze(
                        dim=1).detach()  # B x 1025 x 257 -> B x 1 x 257 x 1025
                    noisy_data = noisy_data.transpose(dim0=1, dim1=2).unsqueeze(
                        dim=1).detach()  # B x 1025 x 257 -> B x 1 x 257 x 1025
                    noisy_feature_data = valid_data[2].unsqueeze(dim=1).detach()  # B x 1 x 1025 x 1799

                if 'MLP' in self.model_name:
                    outputs = generator(noisy_feature_data).transpose(dim0=2, dim1=3)
                else:
                    outputs = generator(noisy_data)
                # L1 loss of Generator
                l1_loss = F.l1_loss(outputs, clean_data)
                # save loss history
                valid_loss_history.append(l1_loss.item())
                # clean-noisy distance
                c_l_dist = F.l1_loss(clean_data, noisy_data)
                valid_bar.set_description(
                    'Epoch{}: validation loss:{:.4f}, clean-noisy-distance:{:.4f}'.format(epoch + 1,
                                                                                          l1_loss.item(),
                                                                                          c_l_dist.item()))
            return np.mean(valid_loss_history)

    def _early_stopping(self, valid_loss_history):
        if len(valid_loss_history) < 4:
            return

        if valid_loss_history[-1] > valid_loss_history[-2] > valid_loss_history[-3]:
            self.early_stopping_patient += 1
            print(f'valid loss increased, patient score is {self.early_stopping_patient} now!')
        else:
            # if not continuous increase of valid loss, reset to 0 again
            self.early_stopping_patient = 0
        return

    def _converge_checking(self, train_loss_history):
        if len(train_loss_history) < self.batch_size * 8:
            return
        # take the last 100 values of loss
        var1 = np.var(train_loss_history[-self.batch_size * 4:])
        var2 = np.var(train_loss_history[-self.batch_size * 8: -self.batch_size * 4])
        if abs(var1 - var2) < self.converge_threshold:
            self.early_stopping_patient += 1
            print(f'training loss converged, patient score is {self.early_stopping_patient} now!')
        else:
            return

    def _test_and_save(self, model, epoch):
        print('Saving test sample and model...')
        with torch.no_grad():
            model.eval()
            test_bar = tqdm(self.test_data_loader, desc='Test model and save generated audios')
            for test_file_name, clean_t, noisy_t in test_bar:
                # calculate phase for sythesis
                # 1 x 16384 -> 16384 -> 257 x 1025
                spec = signal_to_spectrogram(noisy_t.squeeze().numpy())
                phase = get_phase(spec)
                # prepare data to feed model
                test_data = (clean_t, noisy_t)
                test_data = self._prepare_train_data(test_data)
                # only need noisy data
                if self.model_name == 'adversarial_MLP':
                    noisy_data = test_data[2]
                else:
                    noisy_data = test_data[1]

                if self.using_spectrogram:
                    if 'GAN' in self.model_name or 'auto' in self.model_name:
                        # 1 x 1 x 257 x 1025 -> 257 x 1025
                        fake_spec = model(noisy_data).detach().cpu().squeeze().numpy()
                    elif 'MLP' in self.model_name:
                        # 1 x 1025 x 257 -> 1025 x 257 -> 257 x 1025
                        fake_spec = model(noisy_data).detach().cpu().squeeze().numpy()
                        fake_spec = fake_spec.T
                    else:
                        raise NotImplemented

                    # log_power back to magnitude
                    fake_spec = lps_to_mag(fake_spec)
                    # magnitude back to cpmplex
                    fake_spec = magnitude_to_complex(fake_spec, phase)
                    # back to audio signal
                    # 16384
                    fake_speech = librosa.istft(fake_spec, win_length=32, hop_length=16, window='hann')

                else:
                    # 16384
                    fake_speech = model(noisy_data).detach().cpu().squeeze().numpy()

                save_path = os.path.join(f'{self.model_name}_results', 'results', f'{self.criterion_name}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # de-emphasis
                fake_speech = emphasis(fake_speech, emph_coeff=0.95, pre=False)
                # save speech as .wav file
                file_name = os.path.join(save_path, '{}.wav'.format(test_file_name[0].replace('.wav', '')))
                wavfile.write(file_name, 16000, fake_speech)

            # save the model parameters for each epoch
            save_path = os.path.join(f'{self.model_name}_results', 'model')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_path = os.path.join(save_path, f'{self.model_name}-e{epoch}-{self.criterion_name}.pt')
            torch.save(model.state_dict(), model_path)
            print(f'model saved at {model_path}')
            return

    # all training data should go through this method
    def _prepare_train_data(self, data):
        # tensor with batch_size x 16384
        clean_t = data[0].to(self.device)
        noisy_t = data[1].to(self.device)
        # Only True when using 1D GAN model
        if not self.using_spectrogram:
            clean_t = clean_t.unsqueeze(dim=1)
            noisy_t = noisy_t.unsqueeze(dim=1)
            return clean_t, noisy_t

        else:
            # implement STFT
            clean_spec = self._signal_to_spec(clean_t)
            noisy_spec = self._signal_to_spec(noisy_t)
            # get the magnitude as torch tensor
            clean_mag = self._complex_to_mag(clean_spec)
            noisy_mag = self._complex_to_mag(noisy_spec)
            # turn to log_power_spec
            log_clean_mag = self._mag_to_log_mag(clean_mag)
            log_noisy_mag = self._mag_to_log_mag(noisy_mag)
            # this is the case using 2D GAN model
            if 'GAN' in self.model_name or 'simple' in self.model_name or 'auto' in self.model_name:
                # unsqueeze a channel for 2d convolution -> B x 1 x 257 x1025
                return log_clean_mag.unsqueeze(dim=1), log_noisy_mag.unsqueeze(dim=1)
                # return clean_mag.unsqueeze(dim=1), noisy_mag.unsqueeze(dim=1)

            elif 'MLP' in self.model_name:
                # with size B x 257 x 1025 -> B x 1025 x 257
                trans_log_clean_mag = log_clean_mag.transpose(dim0=1, dim1=2)
                trans_log_noisy_mag = log_noisy_mag.transpose(dim0=1, dim1=2)
                # trans_log_clean_mag = clean_mag.transpose(dim0=1, dim1=2)
                # trans_log_noisy_mag = noisy_mag.transpose(dim0=1, dim1=2)

                # check whether in right dimension
                time_len = trans_log_clean_mag.size(1)
                if time_len != 1025:
                    raise ValueError('May select the wrong dim.')

                frame_list = []
                pad = torch.zeros(trans_log_noisy_mag.size(0), 3, 257).to(self.device)
                pad_noisy_mag = torch.cat((pad, trans_log_noisy_mag, pad), dim=1)
                for j in range(time_len):
                    # B x 7 x 257
                    tmp = pad_noisy_mag[:, j:j + 7, :]
                    # B x 1799
                    tmp = tmp.flatten(start_dim=1)
                    # B x 1 X 1799
                    tmp = tmp.unsqueeze(dim=1)
                    frame_list.append(tmp)
                # construct a feature matrix: B x 1025 x 1799
                noisy_feature_input = torch.cat(frame_list, dim=1).to(self.device)
                if self.model_name == 'adversarial_MLP':
                    # B x 1025 x 257 ; B x 1025 x 257; B x 1025 x 1799
                    return trans_log_clean_mag, trans_log_noisy_mag, noisy_feature_input
                # B x 1025 x 257 ; B x 1025 x 1799
                return trans_log_clean_mag, noisy_feature_input
            else:
                raise ValueError('No such model for training')

    def _signal_to_spec(self, t):
        # convert a tensor (B x 16384) to spectrogram
        spec = torch.stft(t,
                          n_fft=512,
                          win_length=32,
                          hop_length=16,
                          window=torch.hann_window(window_length=32))
        # spec size B x 257 x 1025 x 2
        return spec

    def _complex_to_mag(self, spec):
        # convert a torch.complex spectrogram to magnitude spectrogram
        # spec size B x 257 x 1025 x 2
        real = spec[:, :, :, 0]
        img = spec[:, :, :, 1]
        # calculate magnitude
        mag = (real ** 2 + img ** 2).sqrt()
        # mag has a size B x 257 x 1025
        return mag

    def _mag_to_log_mag(self, mag):
        epsilon = torch.tensor(1e-20, device=self.device)  # prevent 'nan' error
        mag = mag + epsilon
        log_mag = torch.log(mag ** 2)
        # log_mag size: B x 257 x 1025
        return log_mag

    def train(self):
        if 'GAN' in self.model_name or self.model_name == 'simple_generator' or self.model_name == 'adversarial_MLP':
            self._train_gan_model()
        elif 'MLP' in self.model_name:
            self._train_mlp_model()
        elif 'auto' in self.model_name:
            self._train_autoencoder()
        else:
            raise NotImplemented

