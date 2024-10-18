# from realcugan_trt.video_encoder import FFMPEG_VideoWriter
from realcugan_trt.video_encoder import VideoEncoderGroup
import pickle
import numpy as np
import ffmpegio
import os
import struct
from multiprocessing import Pool
import multiprocessing
import threading
import cv2
from PIL import Image

frames = np.load('/home/jeremyguo/tmp/frame.npy')
# print(frames.shape, frames.dtype)
# print(frames[0])

def save_48bit_bmp(data):
    """
    保存48位BMP图像
    :param image: numpy数组，形状为(height, width, 3)，数据类型为uint16
    :param filename: 输出文件名
    """
    idx, image = data
    # img = Image.fromarray(image, 'I;16')
    filename = f'/dev/shm/16bit{idx}.tiff'
    # img.save(filename, format="TIFF", compression="tiff_lzw")
    # LZW
    cv2.imwrite(filename, image, [cv2.IMWRITE_TIFF_COMPRESSION, 5])

# from concurrent.futures import ThreadPoolExecutor
import time
# start_time = time.time()
# # MultiThreading Pools
# with Pool(multiprocessing.cpu_count()) as pool:
#     pool.map(save_48bit_bmp, enumerate(frames))
# # np.save('/dev/shm/frames.npy', frames)
# print("Saving Rate = ", len(frames) / (time.time() - start_time))
# os._exit(0)

# start_time = time.time()
# ffmpegio.video.write('/home/jeremyguo/tmp/output2.mp4', 24, frames, show_log=True)
# print("Rate = ", len(frames) / (time.time() - start_time))
# os._exit(0)


start_time = time.time()
N_copy = 4
files = []
for i in range(N_copy):
    filename = f'/dev/shm/output{i}.mp4'
    files.append(filename)
# encoder = FFMPEG_VideoWriter('/home/jeremyguo/tmp/output.mp4', 24, 3840, 2160, ['-c:v', 'libx264', '-preset', 'slower', '-bitrate', '40M'])
encoder = VideoEncoderGroup(files, 24, 3840, 2160, frames.dtype, N_copy*len(frames), ['-c:v', 'hevc_nvenc', '-preset', 'fast', '-bitrate', '10M'])
# encoder = VideoEncoder('/home/jeremyguo/tmp/output.mp4', 24, 3840, 2160, frames.dtype, ['-c:v', 'libx264', '-preset', 'slower', '-bitrate', '40M'])
for i in range(N_copy):
    for frame in frames:
        encoder.encode_frame(frame)
encoder.close()
print("Rate = ", N_copy*len(frames) / (time.time() - start_time))
