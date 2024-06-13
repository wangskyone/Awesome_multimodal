import ffmpeg
import numpy as np
import sys
import random
import cv2
import os
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp


makedirs_lock = mp.Lock()

def create_directory(path):
    with makedirs_lock:
        os.makedirs(path, exist_ok=True)

def wav(path):

    in_path = "/nas_data/WTY/dataset/UCF-101/" + path
    out_path = "/nas_data/WTY/dataset/UCF-101/UCF-WAV/" + path.replace('.avi', '.wav')
    dir_name = os.path.dirname(out_path)

    create_directory(dir_name)
    
    try:
        (ffmpeg
            .input(in_path)
            .output(out_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k',loglevel="quiet",threads='auto')
            .overwrite_output()
            .run(capture_stdout=True)
        )
    except ffmpeg.Error as e:
        return f"Error processing {in_path}: {e.stderr}"

    return f"Processed {in_path} to {out_path}"

    
def frame(path):
    in_path = "/nas_data/WTY/dataset/UCF-101/" + path
    out_path = "/nas_data/WTY/dataset/UCF-101/UCF-Frame/" + path.replace('.avi', '.npy')
    dir_name = os.path.dirname(out_path)

    create_directory(dir_name)

    probe = ffmpeg.probe(in_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    d = probe['format']['duration']
    ss = random.uniform(0, max(float(d) - 4, 0.5))
    ss = round(ss, 1)
    

    # with open("ffmpeg_output.txt", "w") as out, open("ffmpeg_errors.txt", "w") as err:
    #     ffmpeg.run(stream, stdout=out, stderr=err)

    # # 定义ffmpeg管道
    try:
        out, _ = (
            ffmpeg
            .input(in_path, ss=ss - 0.1)  # 从15秒开始
            .trim(duration=4)  # 截取4秒
            .filter('fps', fps=25, round='up')  # 设置fps为25
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', ss=0.1, loglevel="quiet",threads='auto')  # 输出为原始视频格式
            .overwrite_output()
            .run(capture_stdout=True)  # 捕获标准输出
        )
    except ffmpeg.Error as e:
        return f"Error processing {in_path}: {e.stderr}"

    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    np.save(out_path, video)

    return f"Processed {in_path} to {out_path}"


def pre_process(label_dir = '/nas_data/WTY/dataset/UCF-101/Label/'):

    with open(label_dir + 'classInd.txt', 'r') as f:
        lines = f.readlines()
        label2id = {line.strip().split(' ')[1]: int(line.strip().split(' ')[0]) for line in lines}
        id2label = {int(line.strip().split(' ')[0]): line.strip().split(' ')[1] for line in lines}

    trainlist, testlist, labels = [], [], []
    with open(label_dir + 'trainlist.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            path, id = line.strip().split(' ')
            trainlist.append(path)
            labels.append(id)

    with open(label_dir + 'testlist.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            path = line.strip()
            testlist.append(path)

            key = path.split('/')[0]
            if key in label2id:
                labels.append(label2id[key])
            

    assert len(trainlist) + len(testlist) == len(labels)


    # with Pool(10) as pool:
    #     wav_res = list(tqdm(pool.imap(wav, trainlist + testlist), total=len(trainlist) + len(testlist)))
    #     frame_res = list(tqdm(pool.imap(frame, trainlist + testlist), total=len(trainlist) + len(testlist)))


    return label2id, id2label, trainlist, testlist






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
