import argparse
import sys
from .config import Config, default_config, encoders
from .sr_engine import SREngine
from .upcunet2x import RealCUGANUpScaler2x

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

class SRVideoStat:
    def __init__(self):
        self.total_frames = 0
        self.encoded_frames = 0
        self.sred_frames = 0
    
    def getProgress(self):
        if self.total_frames == 0:
            return 0
        return self.sred_frames / self.total_frames

def sr_video(engine:SREngine, inp_file:str, out_file:str, config:Config, verbose=True, stat:SRVideoStat=SRVideoStat()):
    if verbose: print ("Super Resolution started... {} {}".format(inp_file, out_file))
    objVideoreader = VideoFileClip(filename=inp_file)
    w, h = objVideoreader.reader.size
    if w != config.sr_width or h != config.sr_height:
        if verbose: print(f"Input video resolution {w}x{h} is not equal to SR resolution {config.sr_width}x{config.sr_height}")
        return
    fps = objVideoreader.reader.fps
    stat.total_frames = objVideoreader.reader.nframes
    if_audio = objVideoreader.audio

    # Clear Tmp Directory, TODO: dirty codes, modify later
    os.system(f'rm -rf {config.sr_tmp_dir}/*')
    os.system(f'mkdir -p {config.sr_tmp_dir}')

    import pathlib
    output_suffix = pathlib.Path(out_file).suffix
    if output_suffix not in ['.mp4', '.mkv']:
        if verbose: print(f"suffix not in ['.mp4', '.mkv'], got {output_suffix}")
        raise ValueError(f"Output file must be .mp4 or .mkv, but got {output_suffix}")

    # Audio saving & Create Writer
    audio_path = None
    if if_audio:
        audio_path = str(config.sr_tmp_dir / 'audio.wav')
        objVideoreader.audio.write_audiofile(audio_path, logger=None)
        if verbose: print(f"Audio saved to {audio_path}")
    
    import threading
    import time
    import subprocess as sp
    from subprocess import DEVNULL

    encode_clip_number_lock = threading.Lock()
    encode_clip_number_condition = threading.Condition(encode_clip_number_lock)
    encode_clip_number = 0
    encode_clip_list = []
    encoded_frames = 0
    total_encoded_time_ms = 0

    sr_clip_number_lock = threading.Lock()
    sr_clip_number_condition = threading.Condition(sr_clip_number_lock)
    sr_clip_number = 0
    sr_clip_number_frames = {}
    sr_finished = False
    sr_start_time = time.time()
    sr_frames = 0

    def Encoder():
        nonlocal encode_clip_number
        nonlocal encoded_frames
        nonlocal total_encoded_time_ms
        while True:
            with sr_clip_number_condition:
                while encode_clip_number == sr_clip_number and not sr_finished:
                    sr_clip_number_condition.wait()
            if sr_finished and encode_clip_number == sr_clip_number:
                break
            start_time = time.time()
            clip_folder = config.sr_tmp_dir / str(encode_clip_number)
            clip_path = config.sr_tmp_dir / f'{encode_clip_number}{output_suffix}'
            cmd = []

            encoder_params = encoders[config.sr_encoder]
            for param in encoder_params:
                cmd.append(param)
                cmd.append(str(encoder_params[param]))
            
            cmd = ['ffmpeg', '-r', str(fps), '-f', 'image2', '-s', f'{w}x{h}', '-i',
                    f'{clip_folder}/%d.bmp', '-c:v', config.sr_encoder] + cmd + ['-pix_fmt', 'yuv420p', str(clip_path)]
            if sp.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL).wait() != 0:
                print(f"Failed to encode clip {inp_file}")
                os._exit(1)
            
            stat.encoded_frames += sr_clip_number_frames[encode_clip_number]
            encoded_frames += sr_clip_number_frames[encode_clip_number]
            total_encoded_time_ms += (time.time() - start_time) * 1000
            os.system(f'rm -rf {clip_folder}')
            with encode_clip_number_condition:
                encode_clip_number += 1
                encode_clip_number_condition.notify_all()
            
            encode_clip_list.append(clip_path)
    
    def SRThread():
        nonlocal sr_clip_number
        nonlocal sr_finished

        collected_frames = {}

        current_cache_size = 0 # os.path.getsize(file_path)
        current_frame_index = 0
        current_clip_number = 0
        collected_frames[current_clip_number] = 0
        sr_clip_number_frames[current_clip_number] = 0
        os.system(f'mkdir -p {str(config.sr_tmp_dir / str(current_clip_number))}')

        BATCH_SIZE = 32
        frame_in_engine = 0

        def collect(number_of_frames):
            nonlocal current_clip_number
            nonlocal current_frame_index
            nonlocal current_cache_size
            nonlocal frame_in_engine
            nonlocal collected_frames
            nonlocal sr_clip_number_frames
            nonlocal sr_clip_number
            nonlocal sr_finished
            nonlocal sr_frames

            for _ in range(number_of_frames):
                index, output = engine.get()
                sr_frames += 1
                stat.sred_frames += 1
                frame_in_engine -= 1

                path = config.sr_tmp_dir / str(index[0]) / f'{index[1]}.bmp'
                output = cv2.cvtColor(output[0], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(path), output)

                if index[0] == current_clip_number:
                    current_cache_size += os.path.getsize(str(path))
                    if current_cache_size > config.sr_buffer_size:
                        sr_clip_number_frames[current_clip_number] = current_frame_index

                        current_clip_number += 1
                        current_cache_size = 0
                        current_frame_index = 0
                        collected_frames[current_clip_number] = 0
                        sr_clip_number_frames[current_clip_number] = 0
                        os.system(f'mkdir -p {str(config.sr_tmp_dir / str(current_clip_number))}')
                
                collected_frames[index[0]] += 1
                if collected_frames[index[0]] == sr_clip_number_frames[index[0]]:
                    with sr_clip_number_condition:
                        sr_clip_number += 1
                        sr_clip_number_condition.notify_all()

        for frame in objVideoreader.iter_frames():
            engine.push((current_clip_number, current_frame_index), frame)
            current_frame_index += 1
            frame_in_engine += 1

            if frame_in_engine == BATCH_SIZE * 2:
                create_new_frame = collect(BATCH_SIZE)
                if sr_clip_number < current_clip_number - 1:
                    create_new_frame = create_new_frame or collect(BATCH_SIZE)
                if create_new_frame:
                    with encode_clip_number_condition:
                        while encode_clip_number < sr_clip_number - 1:
                            if verbose: print("Waiting for encoder? This is a happy message.")
                            encode_clip_number_condition.wait()
        
        if current_frame_index > 0:
            sr_clip_number_frames[current_clip_number] = current_frame_index

            current_clip_number += 1
            current_cache_size = 0
            current_frame_index = 0
        if frame_in_engine > 0:
            collect(frame_in_engine)
        if sr_clip_number != current_clip_number:
            raise ValueError("BUG: sr_clip_number != current_clip_number")
        with sr_clip_number_condition:
            sr_finished = True
            sr_clip_number_condition.notify_all()  
    sr_thread = threading.Thread(target=SRThread)
    encoder_thread = threading.Thread(target=Encoder)
    sr_thread.start()
    encoder_thread.start()

    if verbose:
        while sr_thread.is_alive() or encoder_thread.is_alive():
            print(f"Encode Rate: {sr_frames / (time.time() - sr_start_time)} fps")
            time.sleep(1)
    
    sr_thread.join()
    encoder_thread.join()

    file_list_path = config.sr_tmp_dir / 'file_list.txt'
    with open(file_list_path, 'w') as f:
        for clip in encode_clip_list:
            f.write(f"file '{clip}'\n")
    tmp_out_file = config.sr_tmp_dir / f'tmp_out{output_suffix}'
    cmd = ['ffmpeg', '-safe', '0', '-loglevel', 'error', '-f', 'concat', '-i', str(file_list_path)]
    if audio_path: cmd += ['-i', audio_path, '-c:a', 'aac']
    else: cmd += ['-c', 'copy']
    cmd.append(str(tmp_out_file))
    if sp.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL).wait() != 0:
        if verbose: print(f"Failed to concat clips {inp_file}")
        raise ValueError("Failed to concat clips")

    def get_subtitles(file_path):
        command = [
            'ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams',
            '-select_streams', 's', file_path
        ]
        import json
        try:
            result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            streams = json.loads(result.stdout).get('streams', [])
            if len(streams) > 0:
                return [stream["index"] for stream in streams]
            return []
        except sp.CalledProcessError as e:
            return []

    subtitles = get_subtitles(inp_file)
    if len(subtitles) > 0:
        subtitle_extract_cmds = ['ffmpeg', '-i', f'{tmp_out_file}']
        for index in subtitles:
            subtitle_extract_cmds += ['-map', f'0:s:{index}', f"{config.sr_tmp_dir}/subtitle_{index}.srt"]
        if sp.Popen(subtitle_extract_cmds, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL).wait() != 0:
            if verbose: print(f"Failed to extract subtitles from {inp_file}")
            raise ValueError("Failed to extract subtitles")
        
        encode_subtitle_cmds = ['ffmpeg', '-i', f'{tmp_out_file}']
        for index in subtitles:
            encode_subtitle_cmds += ['-f', 'srt', '-i', f"{config.sr_tmp_dir}/subtitle_{index}.srt"]
        encode_subtitle_cmds += ['-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text', f'{out_file}']
        if sp.Popen(encode_subtitle_cmds, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL).wait() != 0:
            if verbose: print (f"Failed to encode subtitles to {out_file}")
            raise ValueError("Failed to encode subtitles")
    os.system(f'mv "{tmp_out_file}" "{out_file}"')


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
    if args.old_width is not None:
        config.sr_width = args.old_width
    if args.old_height is not None:
        config.sr_height = args.old_height
    if args.scale is not None:
        config.sr_scale = args.scale
    engine = sr_create_engine(config)
    if args.image:
        sr_image(engine, args.input, args.output, config)
    elif args.video:
        sr_video(engine, args.input, args.output, config)

if __name__ == "__main__":
    main()