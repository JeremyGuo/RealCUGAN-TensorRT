import argparse
import sys
from config import Config, default_config, encoders
from sr_engine import SREngine
from upcunet2x import RealCUGANUpScaler2x

from moviepy.editor import VideoFileClip
import cv2
import os

def sr_create_engine(config:Config, start=True) -> SREngine:
    models = []
    for i in range(config.sr_threads):
        if config.sr_scale == 2:
            model = RealCUGANUpScaler2x(config.sr_modelpath, 
                                        config.sr_half, 
                                        device=config.sr_device,
                                        tile=config.sr_tile,
                                        alpha=config.sr_alpha,
                                        w=config.sr_width, h=config.sr_height,
                                        export_engine=True)
        else:
            print(f"Unsupported scale {config.sr_scale}")
            return None
        models.append(model)
    engine = SREngine(models)
    if start:
        engine.start()
    return engine

def sr_video(engine:SREngine, inp_file:str, out_file:str, config:Config):
    objVideoreader = VideoFileClip(filename=inp_file)
    w, h = objVideoreader.reader.size
    if w != config.sr_width or h != config.sr_height:
        print(f"Input video resolution {w}x{h} is not equal to SR resolution {config.sr_width}x{config.sr_height}")
        return
    fps = objVideoreader.reader.fps
    total_frames = objVideoreader.reader.nframes
    if_audio = objVideoreader.audio

    # Clear Tmp Directory, TODO: dirty codes, modify later
    os.system(f'rm -rf {config.sr_tmp_dir}/*')
    os.system(f'mkdir -p {config.sr_tmp_dir}')

    import pathlib
    output_suffix = pathlib.Path(out_file).suffix
    if output_suffix not in ['.mp4', '.mkv']:
        print(f"suffix not in ['.mp4', '.mkv'], got {output_suffix}")
        raise ValueError(f"Output file must be .mp4 or .mkv, but got {output_suffix}")

    # Audio saving & Create Writer
    audio_path = None
    if if_audio:
        audio_path = str(config.sr_tmp_dir / 'audio.wav')
        objVideoreader.audio.write_audiofile(audio_path, logger=None)
        print(f"Audio saved to {audio_path}")
    
    import threading
    import time
    import subprocess as sp
    from subprocess import DEVNULL

    encode_clip_number_lock = threading.Lock()
    encode_clip_number_condition = threading.Condition(encode_clip_number_lock)
    encode_clip_number = 0
    encoded_frames = 0
    total_encoded_time_ms = 0

    sr_clip_number_lock = threading.Lock()
    sr_clip_number_condition = threading.Condition(sr_clip_number_lock)
    sr_clip_number = 0
    sr_clip_number_frames = {}
    sr_finished = False

    stop_flag = False

    def Encoder():
        nonlocal encode_clip_number
        nonlocal encoded_frames
        nonlocal total_encoded_time_ms
        while True:
            with sr_clip_number_condition:
                while encode_clip_number == sr_clip_number and not stop_flag and not sr_finished:
                    sr_clip_number_condition.wait()
                if stop_flag or sr_finished:
                    return
            start_time = time.time()
            clip_folder = config.sr_tmp_dir / str(encode_clip_number)
            clip_path = config.sr_tmp_dir / f'{encode_clip_number}{output_suffix}'
            cmd = []

            encoder_params = encoders[config.sr_encoder]
            for param in encoder_params:
                cmd.append(param)
                cmd.append(str(encoder_params[param]))
            
            cmd = ['ffmpeg', '-r', str(fps), '-f', 'image2', '-s', f'{w}x{h}', '-i',
                    f'{clip_folder}/%d.png', '-c:v', config.sr_encoder] + cmd + ['-pix_fmt', 'yuv420p', str(clip_path)]
            if sp.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL).wait() != 0:
                print(f"Failed to encode clip {inp_file}")
                os._exit(1)
            
            encoded_frames += sr_clip_number_frames[encode_clip_number]
            total_encoded_time_ms += (time.time() - start_time) * 1000
            os.system(f'rm -rf {clip_folder}')
            with encode_clip_number_condition:
                encode_clip_number += 1
                encode_clip_number_condition.notify_all()
    
    def SRThread():
        sr_clip_folder = config.sr_tmp_dir / str(sr_clip_number)
        os.system(f'mkdir -p {sr_clip_folder}')

        current_cache_size = 0 # os.path.getsize(file_path)
        for frame in objVideoreader.iter_frames():
            with sr_clip_number_lock:
                pass

def sr_image(engine:SREngine, inp_file:str, out_file:str, config:Config):
    img = cv2.imread(inp_file)
    if img.shape != (config.sr_height, config.sr_width, 3):
        print(f"Input image resolution {img.shape} is not equal to SR resolution {config.sr_width}x{config.sr_height}")
        return
    img = img.reshape(1, config.sr_height, config.sr_width, 3)
    engine.push(0, img)
    print("Super Resolution started...")
    _, result = engine.get()
    print("Super Resolution done, saving...")
    cv2.imwrite(out_file, result[0])
    print("Super Resolution done, cleanup...")
    engine.stop()

def main():
    parser = argparse.ArgumentParser(description="Super Resolution Tool deep optimized for RealCUGAN.")
    parser.add_argument('-i', '--input', required=True, help="Input file path")
    parser.add_argument('-o', '--output', required=True, help="Output file path")
    parser.add_argument('--old_width', type=int, help="Old width of the video")
    parser.add_argument('--old_height', type=int, help="Old height of the video")
    parser.add_argument('--image', action='store_true', help="Super resolution for image")
    parser.add_argument('--video', action='store_true', help="Super resolution for video")
    parser.add_argument('--scale', type=int, help="Super resolution scale")
    args = parser.parse_args()

    if args.image and args.video:
        print("Please choose either image or video super resolution")
        sys.exit(1)
    
    config = default_config
    
    if args.image:
        if args.old_width is not None:
            config.sr_width = args.old_width
        if args.old_height is not None:
            config.sr_height = args.old_height
        if args.scale is not None:
            config.sr_scale = args.scale
        engine = sr_create_engine(config)
        sr_image(engine, args.input, args.output, config)

if __name__ == "__main__":
    main()