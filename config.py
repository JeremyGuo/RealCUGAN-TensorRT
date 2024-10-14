from pathlib import Path

class Config:
    def __init__(self):
        self.sr_threads=2
        self.sr_scale=2
        self.sr_modelpath= Path(__file__).parent / 'weights/pro-conservative-up2x.pth'
        self.sr_tile=5
        self.sr_alpha=1
        self.sr_width=1920
        self.sr_height=1080
        self.sr_half=True
        self.sr_device='cuda:0'

        # ONLY FOR VIDEO SR and can be dynamically changed
        self.sr_encoder='libx264'
        self.sr_tmp_dir=Path.home() / "tmp"
        self.sr_encode_params = ['-crf', '21']
        self.sr_buffer_size = '1G'

default_config = Config()

encoders = {
    'hevc_nvenc': {
        '-preset': 'slow',
        '-bitrate': '40M',
    },
    'libx264' : {
        '-preset': 'slower',
        '-bitrate': '40M',
    }
}