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


class DataProcessor:
    def __init__(self, train_audio_transforms, valid_audio_transforms, sp_transform_e=None, sp_transform_c=None):
        """
        Initializes the DataProcessor class.

        Args:
            train_audio_transforms (callable): Transformations for training data.
            valid_audio_transforms (callable): Transformations for validation/test data.
            sp_transform_e (callable, optional): SentencePiece transformer for English.
            sp_transform_c (callable, optional): SentencePiece transformer for another language.
        """
        self.train_audio_transforms = train_audio_transforms
        self.valid_audio_transforms = valid_audio_transforms
        self.sp_transform_e = sp_transform_e
        self.sp_transform_c = sp_transform_c

    def data_processing(self, data, data_type="train"):
        """
        Processes data for training or validation/test.

        Args:
            data (list): Input data containing waveform and utterance information.
            data_type (str): 'train' or 'valid'/'test'.

        Returns:
            tuple: Processed spectrograms, labels, input_lengths, label_lengths.
        """
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []

        for (waveform, _, utterance, _, _, _) in data:
            if data_type == 'train':
                spec = self.train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            elif data_type in ['test', 'valid']:
                spec = self.valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                raise ValueError("data_type must be 'train', 'valid', or 'test'.")
            spectrograms.append(spec)
            label = torch.Tensor(self.sp_transform_e.text_to_int(utterance))
            labels.append(label)
            input_lengths.append(spec.shape[0])
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths

    def data_processing_c(self, data, data_type="train"):
        """
        Processes data with a different SentencePiece transformer.

        Args:
            data (list): Input data containing waveform and utterance information.
            data_type (str): 'train' or 'valid'/'test'.

        Returns:
            tuple: Processed spectrograms, labels, input_lengths, label_lengths.
        """
        spectrograms = []
        labels = []
        input_lengths = []
        label_lengths = []

        for (waveform, utterance) in data:
            if data_type == 'train':
                spec = self.train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            elif data_type in ['test', 'valid']:
                spec = self.valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            else:
                raise ValueError("data_type must be 'train', 'valid', or 'test'.")
            spectrograms.append(spec)
            label = torch.Tensor(self.sp_transform_c.text_to_int(utterance))
            labels.append(label)
            input_lengths.append(spec.shape[0])
            label_lengths.append(len(label))

        spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return spectrograms, labels, input_lengths, label_lengths
