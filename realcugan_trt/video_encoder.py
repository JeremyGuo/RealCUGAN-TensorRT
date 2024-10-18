import subprocess
import numpy as np
from moviepy.config import FFMPEG_BINARY
from subprocess import DEVNULL
from typing import Sequence
import threading
import multiprocessing

def VideoEncoderProc(params, queue: multiprocessing.Queue, n_frames):
    encoder = VideoEncoder(*params)
    for i in range(n_frames):
        frame = queue.get()
        encoder.encode_frame(frame)
    encoder.close()

class VideoEncoderGroup:
    def __init__(self, output_filenames:Sequence[str], fps, width, height, dtype, n_frames, encoder_params):
        self.output_filenames = output_filenames
        self.encoder_params = [(output_filename, fps, width, height, dtype, encoder_params) for output_filename in output_filenames]
        self.queues : Sequence[multiprocessing.Queue] = [multiprocessing.Queue(128) for _ in range(len(output_filenames))]
        self.counters = [n_frames // len(output_filenames) for _ in range(len(output_filenames))]
        self.counters[-1] += n_frames % len(output_filenames)
        self.n_injected_frames = 0
        self.n_current_encoder = 0
        self.processes = [multiprocessing.Process(target=VideoEncoderProc, args=(param, queue, n_frames)) for param, queue, n_frames in zip(self.encoder_params, self.queues, self.counters)]
        for process in self.processes:
            process.start()

    def encode_frame(self, frame):
        while self.n_current_encoder < len(self.output_filenames) and self.counters[self.n_current_encoder] == self.n_injected_frames:
            self.n_injected_frames -= self.counters[self.n_current_encoder]
            self.n_current_encoder += 1
        if self.n_current_encoder == len(self.output_filenames):
            raise RuntimeError("All encoders are closed.")
        self.queues[self.n_current_encoder].put(frame)
        self.n_injected_frames += 1
    
    def close(self):
        for process in self.processes:
            process.join()

class VideoEncoder:
    def __init__(self, output_filename, fps, width, height, dtype, encoder_params):
        self.output_filename = output_filename
        self.fps = fps
        self.width = width
        self.height = height
        self.process = None
        self.dtype = dtype

        pix_fmt = "yuv420p"
        if dtype == np.uint16:
            pix_fmt = "yuv420p10le"

        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',  # size of one frame
            '-pix_fmt', 'rgb48le',  # input pixel format
            '-r', str(self.fps),  # frames per second
            '-i', '-',  # The input comes from a pipe
            '-an',  # Tells FFMPEG not to expect any audio
            '-pix_fmt', pix_fmt
        ] + encoder_params + [ self.output_filename ]

        print(command)

        # Change PIPE buffer size to 2GB
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        # self.process.start()

    def encode_frame(self, frame):
        if self.process is None:
            raise RuntimeError("Encoder not started. Call start() first.")

        # Ensure the frame is in the correct shape (height, width, channels)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Frame shape should be ({self.height}, {self.width}, 3)")

        # Ensure the frame is uint16
        if frame.dtype != self.dtype:
            raise ValueError("Frame should be uint16")

        # Remove the batch dimension and convert to little-endian
        if self.dtype == np.uint16:
            frame = frame.astype('<u2')

        # Write the frame to the FFmpeg process
        self.process.stdin.write(frame.tobytes())
        self.process.stdin.flush()

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
            self.process = None