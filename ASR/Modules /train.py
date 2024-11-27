from numpy.random.mtrand import beta
import torch
import torch.nn as nn
import torch.optim as optim
from model import conformer
import torchvision
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from ctcdecode import CTCBeamDecoder
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
from wer import calculate_wer
import os
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import time
from data import LibriSpeechDataset, ASR
from utils import InfoNCE
from decode import SentencePieceTransform
###############  hyper-parameters ############

seed = np.random.seed(42)


def get_audio_transforms():
  time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
  train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
    *time_masks,
  )

  valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)

  return train_audio_transform, valid_audio_transform

train_audio_transforms, valid_audio_transforms = get_audio_transforms()


def load(split, batch_size, workers=0, augmentation=False):
    """
    Args:
        split (string): Which of the subset of data to take. One of 'train', 'dev' or 'test'.
        batch_size (integer): Batch size.
        workers (integer): How many subprocesses to use for data loading.
        augmentation (bool): Apply SpecAugment to training data or not.

    Returns:
        loader (DataLoader): A DataLoader can generate batches of (FBANK features, FBANK lengths, label sequence).
    """
    assert split in ['train', 'dev', 'test']

    dataset = ASR(split, augmentation)
    # print(dataset)
    print ("%s set size:"%split.upper(), len(dataset))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=lambda x: data_processing_c(x, split),
                        num_workers=workers,
                        pin_memory=True)
    return loader



def loss_F(parameters):
    return sum(torch.linalg.norm(w) ** 2 for w in parameters)


def get_audio_files_flac(data_dir):
    return [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.lower().endswith('.flac')]

def get_audio_files_wav(data_dir):
    return [os.path.join(root, file) for root, dirs, files in os.walk(data_dir) for file in files if file.lower().endswith('.wav')]

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 5e-4
    batch_size = 16
    epochs = 100
    moo_method = 'MoDo'
    modo_gamma = 0.1
    modo_rho = 0.0

    hparams = {
        "n_class": 5000,
        "n_feats": 80,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }

    # Prepare device and random seed
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargsd = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}

    # Create data directory if it doesn't exist
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    # Load train and test datasets for English
    splits = ["train-clean-100", "train-clean-360", "train-clean-500"]
    datasets = [LIBRISPEECH("./data", url=split, download=True) for split in splits]
    combined_dataset = ConcatDataset(datasets)
    test_dataset = LIBRISPEECH("./data", url="test-clean", download=True)

    # Prepare DataLoaders
    train_loader_e = DataLoader(
        dataset=combined_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=lambda x: data_processing(x, 'train'),
        **kwargsd
    )
    test_loader_e = DataLoader(
        dataset=test_dataset,
        batch_size=50,
        shuffle=False,
        collate_fn=lambda x: data_processing(x, 'valid'),
        **kwargsd
    )

  # Load train and test datasets for Chinese

    train_loader_c = load('train', batch_size)
    test_loader_c = load('test', batch_size)


    # Model setup
    model = conformer(num_classes=hparams['n_class'], input_dim=hparams['n_feats'], num_encoder_layers=8)
    model = nn.DataParallel(model)
    model.to(device)

    # Print model information
    print('Num Model Parameters:', sum(param.nelement() for param in model.parameters()))
    num_param, num_param_layer = get_layer_params(model)

    # Optimizer, scheduler, and loss function setup
    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['learning_rate'],
        steps_per_epoch=int(len(train_loader_e)),
        epochs=hparams['epochs'],
        anneal_strategy='linear',
    )
    criterion = nn.CTCLoss(blank=0).to(device)
    loss_dict = {'eng': criterion, 'chn': criterion}

    # Multi-objective optimization settings
    num_tasks = 2
    multi_grad_fn = {'MoDo': grad_modo}
    modo_kwargs = {'lambd': torch.ones(num_tasks) / num_tasks, 'gamma': modo_gamma, 'rho': modo_rho}
    kwargs = {'MoDo': modo_kwargs}

    # Pre-training setup
    data_dir_c = "./data_aishell/wav"
    data_dir_e = "./data/LibriSpeech"
    audio_files = get_audio_files_wav(data_dir_c) + get_audio_files_flac(data_dir_e)
    waveform_length, context_length, future_length, negative_waveform_length = 16000, 20, 12, 12
    train_dataset2 = LibriSpeechDataset(audio_files, waveform_length, context_length, future_length, negative_waveform_length)
    train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True)
    pre_optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    pre_scheduler = optim.lr_scheduler.OneCycleLR(
        pre_optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_loader2)),
        epochs=hparams['epochs'],
        anneal_strategy='linear',
    )
    pre_criterion = InfoNCE()

    # Adaptive gamma settings
    gamma_max, gamma_init, gamma_argmax_step = 1, 0, 500
    gam = max(gamma_init, gamma_max)
    step_gam = (gamma_max - gamma_init) / gamma_argmax_step

    # Training loop
    train_loss, test_loss, test_loss_c, wer, wer_c = [], [], [], [], []
    best_test_loss = float('inf')

    for epoch in range(1, epochs + 1):
        grad_list = []
        loss_list = [0] * num_tasks
        gam = min(gamma_max, gam)

        # Training
        train_loss_epoch = train(model, device, train_loader_e, train_loader_c, criterion, optimizer, grad_list, loss_list, multi_grad_fn,
                        kwargs, epoch, loss_dict,train_loader2,pre_criterion,pre_optimizer,gam)
        
        gam += step_gam

        # Testing
        test_loss_eng, wer_en = test(model, device, test_loader_e, criterion, epoch, 1)
        test_loss_ch, wer_ch = test(model, device, test_loader_c, criterion, epoch, 2)
        scheduler.step()
        pre_scheduler.step()

        # Track losses and WERs
        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_eng)
        test_loss_c.append(test_loss_ch)
        wer.append(wer_epoch)
        wer_c.append(wer_c_epoch)

        # Save best model
        if test_loss_en < best_test_loss or test_loss_ch < best_test_loss:
            best_test_loss = min(test_loss_epoch, test_loss_c_epoch)
            torch.save(model.state_dict(), './conformer/modo_just960.pth')
            print('Model saved!')

        print(f'Best test loss: {best_test_loss}')

        # Log results
        with open('modo_bilevel_960.txt', 'a') as file:
            file.write(f'Test Loss (English) Epoch {epoch}: {test_loss_epoch:.4f}\n')
            file.write(f'Test Loss (Chinese) Epoch {epoch}: {test_loss_c_epoch:.4f}\n')
            file.write(f'Gamma Epoch {epoch}: {gam:.4f}\n')

    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
