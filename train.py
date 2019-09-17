from model_trainer import ModelTrainer
import argparse
model_names = ['ResSEGAN_trained_by_signal', 'ResSEGAN_trained_by_spectrogram', 'MLP', 'simple_generator',
               '1D_auto-encoder', '2D_auto-encoder', 'simple_auto-encoder', 'adversarial_MLP']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Audio Enhancement')
    parser.add_argument('--model_name', default=0, type=int, help='choose model')
    parser.add_argument('--batch_size', default=512, type=int, help='train batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--num_epochs', default=30, type=int, help='train epochs number')
    parser.add_argument('--criterion', default='BCE', type=str, help='Loss function type')
    parser.add_argument('--num_GPU', default=1, type=int, help='train GPU number')
    parser.add_argument('--num_workers', default=8, type=int, help='num of workers')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    parser.add_argument('--using_l1', default=False, type=bool, help='whether using L1 restriction for GAN')
    parser.add_argument('--using_spectrogram', default=False, type=bool, help='whether use spectrogram as feature')
    parser.add_argument('--converge_threshold', default=0.001, type=float, help='for converge checking')
    parser.add_argument('--pin_memory', default=False, type=bool, help='for dataloader pin_memory')
    parser.add_argument('--start_GPU', default=0, type=int, help='start GPU')
    parser.add_argument('--using_simple_g', default=False, type=bool, help='whether use simple g net')
    opt = parser.parse_args()

    train_config = {'model_name': model_names[opt.model_name],
                    'batch_size': opt.batch_size,
                    'lr': opt.lr,
                    'num_epochs': opt.num_epochs,
                    'criterion': opt.criterion,
                    'num_GPU': opt.num_GPU,
                    'num_workers': opt.num_workers,
                    'optimizer': opt.optimizer,
                    'using_l1': opt.using_l1,
                    'using_spectrogram': opt.using_spectrogram,
                    'converge_threshold': opt.converge_threshold,
                    'pin_memory': opt.pin_memory,
                    'start_GPU': opt.start_GPU,
                    'using_simple_g': opt.using_simple_g}

    if opt.model_name == 0:
        # res_SEGAN_signal
        pass
    elif opt.model_name == 1:
        # res_SEGAN spec
        train_config['using_spectrogram'] = True

    elif opt.model_name == 2:
        # MLP
        train_config['using_spectrogram'] = True
        train_config['criterion'] = 'MSE'

    elif opt.model_name == 3:
        # simple generator
        train_config['using_simple_g'] = True
        train_config['criterion'] = 'MSE'

    elif opt.model_name == 4:
        # 1D auto encoder
        train_config['criterion'] = 'MSE'

    elif opt.model_name == 5:
        # 2D auto encoder
        train_config['criterion'] = 'MSE'
        train_config['using_spectrogram'] = True

    elif opt.model_name == 6:
        # simple auto encoder
        train_config['criterion'] = 'MSE'

    elif opt.model_name == 7:
        # adversarial MLP
        train_config['using_spectrogram'] = True

    trainer = ModelTrainer(**train_config)
    trainer.train()



