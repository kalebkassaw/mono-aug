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

#TODO: def save_map(fn)

assert args.i is not None, "Please specify an input file location using -i followed by a file location."
print('Wiggle augmentation enabled.')
print('Maximum jitter magnitude: %i' % args.dev)
print('Input image(s) location:  %s' % args.i)
print('Output image(s) location: %s' % args.o)

augmentation = augs.Wiggle(gray=args.rgb_mode)

if args.o is None:
    try:
        os.mkdir('sample_images')
    except FileExistsError:
        pass
    save_loc = os.cwd + os.sep + 'sample_images'
else:
    save_loc = args.o + os.sep + 'sample_images'

if os.path.isdir(args.i):
    for fn in os.listdir(args.i):
        augmentation(x, )