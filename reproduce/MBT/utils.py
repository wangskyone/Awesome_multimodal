import ffmpeg
import numpy as np
import sys
import random
import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import torchaudio
import torch

makedirs_lock = mp.Lock()

def create_directory(path):
    with makedirs_lock:
        os.makedirs(path, exist_ok=True)


def wav_frame(path):

    in_path = "/nas_data/WTY/dataset/UCF-101/" + path

    probe = ffmpeg.probe(in_path)

    if not has_audio(path):
        return f"Processed {in_path} has no audio"


    frame_out_path = "/nas_data/WTY/dataset/UCF-101/UCF-Frame/" + path.replace('.avi', '.npy')
    frame_dir_name = os.path.dirname(frame_out_path)


    wav_out_path = "/nas_data/WTY/dataset/UCF-101/UCF-WAV/" + path.replace('.avi', '.wav')
    wav_dir_name = os.path.dirname(wav_out_path)

    create_directory(wav_dir_name)
    create_directory(frame_dir_name)


    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    d = probe['format']['duration']
    ss = random.uniform(max(float(d) - 4, 0.5) - 0.1, max(float(d) - 4, 0.5))
    ss = float(format(ss, '.1f'))

    # try:
    (ffmpeg
        .input(in_path, ss=ss, t=4)
        .output(wav_out_path, format='wav', acodec='pcm_s16le', loglevel="quiet" ,ac=1, ar='16k')
        .overwrite_output()
        .run(capture_stdout=True)
    )
    # except ffmpeg.Error as e:
    #     print(f"Error processing {in_path}: {e.stderr}")
    #     return f"Error processing {in_path}: {e.stderr}"

    # try:
    out, _ = (
        ffmpeg
        .input(in_path, ss=ss, t=4)  # 从15秒开始
        .filter('fps', fps=25, round='up')  # 设置fps为25
        .output('pipe:', format='rawvideo', pix_fmt='rgb24',loglevel="quiet" ,ss=0.1, threads='auto')  # 输出为原始视频格式
        .overwrite_output()
        .run(capture_stdout=True)  # 捕获标准输出
    )
    # except ffmpeg.Error as e:
    #     print(f"Error processing {in_path}: {e.stderr}")
    #     return f"Error processing {in_path}: {e.stderr}"

    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    np.save(frame_out_path, video)

    return f"Processed {in_path} to {frame_out_path}"


def has_audio(path):
    in_path = "/nas_data/WTY/dataset/UCF-101/" + path

    probe = ffmpeg.probe(in_path)

    for stream in probe['streams']:
        if stream['codec_type'] == 'audio':
            return True
    
    return False



def pre_process(label_dir = '/nas_data/WTY/dataset/UCF-101/Label/'):

    with open(label_dir + 'classInd.txt', 'r') as f:
        lines = f.readlines()
        label2id = {line.strip().split(' ')[1]: int(line.strip().split(' ')[0]) for line in lines}
        id2label = {int(line.strip().split(' ')[0]): line.strip().split(' ')[1] for line in lines}

    trainlist, testlist, labels = [], [], []
    with open(label_dir + 'trainlist.txt', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            path, id = line.strip().split(' ')
            trainlist.append(path)
            labels.append(id)

    with open(label_dir + 'testlist.txt', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            path = line.strip()
            testlist.append(path)
            key = path.split('/')[0]
            if key in label2id:
                labels.append(label2id[key])
            

    assert len(trainlist) + len(testlist) == len(labels)

    #"BaseballPitch/v_BaseballPitch_g08_c02.avi"
    # wav_frame(trainlist[0])
    with Pool(10) as pool:
        wav_frame_res = list(tqdm(pool.imap(wav_frame, trainlist + testlist), total=len(trainlist) + len(testlist)))


def get_data(label_dir = '/nas_data/WTY/dataset/UCF-101/Label/', mode='train'):
    with open(label_dir + 'classInd.txt', 'r') as f:
        lines = f.readlines()
        label2id = {line.strip().split(' ')[1]: int(line.strip().split(' ')[0]) for line in lines}
        id2label = {int(line.strip().split(' ')[0]): line.strip().split(' ')[1] for line in lines}

    data_list, labels = [], []

    if mode == "train":
        with open(label_dir + 'trainlist.txt', 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                path, id = line.strip().split(' ')
                data_list.append(path)
                labels.append(id)
    else:
        with open(label_dir + 'testlist.txt', 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                path = line.strip()
                data_list.append(path)
                key = path.split('/')[0]
                if key in label2id:
                    labels.append(label2id[key])

    
    new_list, new_labels = [], []

    for path, label in zip(data_list, labels):
        wav_out_path = "/nas_data/WTY/dataset/UCF-101/UCF-WAV/" + path.replace('.avi', '.wav')
        if os.path.exists(wav_out_path):
            new_list.append(path)
            new_labels.append(label)

    print(f"Total {len(new_list)} files found in {mode} mode")
    return new_list, new_labels
            

def cal_spectrogram_mean_std(wav_list):
    specs = []
    for wav_path in tqdm(wav_list):
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        waveform = waveform[0]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=128)(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        specs.append(spec)
    
    
    specs = torch.cat(specs, dim=1)
    spec_mean = torch.mean(specs, dim=1)
    spec_std = torch.std(specs, dim=1)

    torch.save(spec_mean, 'spec_mean.pt')
    torch.save(spec_std, 'spec_std.pt')

    return spec_mean, spec_std






# audio = (ffmpeg
#             .input("/nas_data/WTY/dataset/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g25_c07.avi")
#             .audio
#             .output('-', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
#             .run_async(pipe_stdout=True)
#             )

# out, _ = (ffmpeg
#     .input("/nas_data/WTY/dataset/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g25_c07.avi")
#     .output('test.wav', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
#     .run(capture_stdout=True)
# )


# probe = ffmpeg.probe("/nas_data/WTY/dataset/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g17_c03.avi")


# video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
# width = int(video_info['width'])
# height = int(video_info['height'])

# # # 定义ffmpeg管道
# out, _ = (
#     ffmpeg
#     .input("/nas_data/WTY/dataset/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g17_c03.avi", ss=0.5 - 0.1)  # 从15秒开始
#     .trim(duration=4)  # 截取4秒
#     .filter('fps', fps=25, round='up')  # 设置fps为25
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24', ss=0.1)  # 输出为原始视频格式
#     .run(capture_stdout=True)  # 捕获标准输出
# )


# video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

# for i, frame in enumerate(video):
#     # 对每一帧进行处理
#     # 例如，你可以用OpenCV显示或保存每一帧
#     # import cv2
#     cv2.imwrite(f"frame_{i:04d}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# d = probe['format']['duration']

# ss = random.uniform(0, max(float(d) - 4, 0.1))

# ffmpeg -ss 15 -i input_video.mp4 -t 4 -vf fps=25 -qscale:v 2 image_%04d.jpg
# use python to run the ffmpeg command

          
# import os
# import subprocess

# # 输入视频文件
# input_file = "/nas_data/WTY/dataset/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g17_c03.avi"

# # 随机选取的起始时间(单位:秒)
# start_time = ss

# # 截取时长(单位:秒)
# duration = 4

# # 输出帧率
# fps = 25

# # 输出图像文件名模板
# output_pattern = "image_%04d.jpg"

# # 构建ffmpeg命令
# cmd = f"ffmpeg -ss {start_time} -i {input_file} -t {duration} -vf fps={fps} -qscale:v 2 {output_pattern}"

# # 运行ffmpeg命令
# subprocess.run(cmd, shell=True)
