import argparse
import augs
import os
import tqdm
import numpy as np

desc = "Augment an image with a randomly-generated monotonic function of the image."
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-i', help='input file location', type=str, default=None)
parser.add_argument('-o', help='output file location', type=str, default=None)
parser.add_argument('--dev', help='maximum jitter magnitude', type=int, default=20)
parser.add_argument('--save-map', help='save mapping as NumPy array', action='store_true')
parser.add_argument('--rgb-mode', help='(experimental) use RGB images in augmentation', action='store_true')
args = parser.parse_args()

assert args.i is not None, "Please specify an input file location using -i followed by a file location."
print('Wiggle augmentation enabled.')
print('Maximum jitter magnitude: %i' % args.dev)
print('Input image(s) location:  %s' % args.i)
print('Output image(s) location: %s' % args.o)

augmentation = augs.Wiggle(gray=args.rgb_mode)

def get_name(fn):
    im = fn.split(os.sep)[-1]
    extns = ['.jpg', '.jpeg', '.png', '.bmp']
    [im.replace(e, '') for e in extns]
    return im

if args.o is None:
    counter += 1
    try:
        appx = str(counter) if counter > 1 else ''
        os.mkdir('sample_images' + appx)
        os.mkdir('sample_maps' + appx)
    except FileExistsError:
        pass
    save_loc = os.cwd + os.sep + 'sample_images'
    map_loc  = os.cwd + os.sep + 'sample_maps'
else:
    save_loc = args.o + os.sep + 'sample_images'
    map_loc  = args.o + os.sep + 'sample_maps'

if os.path.isdir(args.i):
    for fn in os.listdir(args.i):
        augmentation()