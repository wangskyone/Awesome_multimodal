import os
import pandas as pd
from PIL import Image
import numpy as np
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tv
import torchaudio.transforms as Ta
import torchaudio
from utils import get_data
from transformers import WhisperFeatureExtractor


class AV_Dataset(Dataset):
    def __init__(self, data, audio_dir, image_dir, labels, spec_mean, spec_std, num_images_per_clip=8):
        super(AV_Dataset, self).__init__()


        self.data = data
        self.audio_dir = audio_dir
        self.img_dir = image_dir
        self.labels = labels
        self.num_images_per_clip = num_images_per_clip

        self.visual_transforms = Tv.Compose([
            # input image (224x224) by default
            Tv.ToPILImage(),
            Tv.Resize((224, 224)),
            Tv.ToTensor(),
            Tv.ConvertImageDtype(torch.float32),
            # normalize to imagenet mean and std values
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sampling_frequency = 16000
        self.mel = Ta.MelSpectrogram(
                      sample_rate=16000,
                      n_fft=400,
                      win_length=400,
                      hop_length=160,
                      n_mels=128,
                      center=True
                  )
                  
        self.a2d = Ta.AmplitudeToDB()
        # mean and std already calculated.
        self.spec_mean = spec_mean.unsqueeze(1)
        self.spec_std = spec_std.unsqueeze(1)
        self.max_lens = 400

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # select one clip
        clip_name = self.data[idx]

        # load the audio file with torch audio
        audio_path = self.audio_dir + clip_name.replace('.avi', '.wav')
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        # use mono audio instead os stereo audio (use left by default)
        waveform = waveform[0]

        # resample
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)

        # normalize raw waveform
        waveform = (waveform - torch.mean(waveform)) / (torch.std(waveform) + 1e-6)
        # generate mel spectrogram and convert amplitude to decibels
        spectrogram = self.a2d(self.mel(waveform))
        # normalize spectrogram.
        # spectrogram = (spectrogram - self.spec_mean) / self.spec_std
        spectrogram = spectrogram.type(torch.float32)

        # def pad_and_truncate(tensor, max_len):
        #     if tensor.shape[1] > max_len:
        #         tensor = tensor[:, :max_len]
        #     else:
        #         pad = torch.zeros(tensor.shape[0], max_len - tensor.shape[1])
        #         tensor = torch.cat([tensor, pad], dim=1)
        #     return tensor
        
        # spectrogram = pad_and_truncate(spectrogram, self.max_lens)


        # load images
        file_path = self.img_dir + clip_name.replace('.avi', '.npy')
        imgs = np.load(file_path)

        # resampling indices
        target_frame_idx = np.linspace(0, len(imgs)-1 , num=self.num_images_per_clip, dtype=int)
        rgb_frames = []
        for i in target_frame_idx:
            img = imgs[i]
            img = self.visual_transforms(img)
            rgb_frames.append(img)
        
        rgb_frames = torch.stack(rgb_frames, dim=0)


        # assign integer to labels
        label = int(self.labels[idx])
        
        return spectrogram, rgb_frames, label




if __name__ == "__main__":
    data_list, labels = get_data()
    spec_mean = torch.load('spec_mean.pt')
    spec_std = torch.load('spec_std.pt')
    train = AV_Dataset(data_list, "/nas_data/WTY/dataset/UCF-101/UCF-WAV/", "/nas_data/WTY/dataset/UCF-101/UCF-Frame/", 
                       labels, spec_mean, spec_std)

    data_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, pin_memory=True)

    for i, (spec, rgb, label) in enumerate(data_loader):
        print(spec.shape, rgb.shape, label)
        if i == 0:
            break