from realcugan_trt.sr_tool import sr_create_engine, sr_image, sr_video
from realcugan_trt.config import default_config
import argparse
import sys

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
    if not args.image and not args.video:
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
    engine.stop()

if __name__ == "__main__":
    main()