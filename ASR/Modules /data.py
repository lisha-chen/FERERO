from numpy.random.mtrand import beta
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import torch.utils.data as data
import torchaudio
import os
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader


class LibriSpeechDataset(Dataset):
    def __init__(self, audio_files, waveform_length, context_length, future_length, negative_waveform_length):
        self.audio_files = audio_files
        self.waveform_length = waveform_length
        self.context_length = context_length
        self.future_length = future_length
        self.negative_waveform_length = negative_waveform_length

    def __len__(self):
        return len(self.audio_files)

    def load_waveform(self, audio_path, waveform_length):
        waveform, _ = torchaudio.load(audio_path)
        if waveform.size(1) > waveform_length:
            start_idx = random.randint(0, waveform.size(1) - waveform_length)
            waveform = waveform[:, start_idx: start_idx + waveform_length]
        else:
            pad_length = waveform_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self.load_waveform(audio_path, self.waveform_length)

        # Generate context waves
        start_idx = random.randint(0, self.waveform_length - self.context_length - self.future_length)
        context = waveform[:, start_idx: start_idx + self.context_length]

        # Generate future samples
        future = waveform[:, start_idx + self.context_length: start_idx + self.context_length + self.future_length]

        # Generate negative sample
        negative_idx = random.randint(0, len(self.audio_files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.audio_files) - 1)

        negative_audio_path = self.audio_files[negative_idx]
        negative_waveform = self.load_waveform(negative_audio_path, self.negative_waveform_length)

        negative_sample = negative_waveform

        # Return context, future, negative sample, and waveform length
        return context, future, negative_sample, context.size(1)

class ASR(Dataset):
    """
    Stores a Pandas DataFrame in __init__, and reads and preprocesses examples in __getitem__.
    """
    def __init__(self, split, augmentation):
        """
        Args:
            augmentation (bool): Apply SpecAugment to training data or not.
        """
        if split.upper()=='TRAIN':
            file_path = '/media/chenlab2/hdd5/saif/asr/conformer/TRAIN.csv'
            self.df = pd.read_csv(file_path)
            
        if split.upper()=='TEST':
            self.df = pd.read_csv('/media/chenlab2/hdd5/saif/asr/conformer/TEST.csv')
        self.augmentation = (augmentation and (split.upper() == 'TRAIN'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
            x (torch.FloatTensor, [seq_length, dim_features]): The FBANK features.
            y (torch.LongTensor, [n_tokens]): The label sequence.
        """
        x, y = self.df.iloc[idx]
        x, sample_rate = torchaudio.load(x)


        return x, y

