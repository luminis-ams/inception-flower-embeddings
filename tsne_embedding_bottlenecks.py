import argparse
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys

import numpy as np
import re
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from sklearn.manifold import TSNE

def imscatter(x, y, image_path, ax, color='black'):
    image = Image.open(os.path.join(image_path))
    image.thumbnail((42, 42))
    image = ImageOps.expand(image, border=3, fill=color)
    im = OffsetImage(image)
    x, y = np.atleast_1d(x, y)
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    ax.add_artist(ab)

def vec_from_string(path):
    with open(path, 'r') as f:
        return np.fromstring(f.readline(), dtype=float, sep=',')

def main(argv):

    parser = argparse.ArgumentParser(description='Do a T-SNE visualisation')
    parser.add_argument('image_dir')
    parser.add_argument('--bottleneck_dir', default='/tmp/bottleneck')
    parser.add_argument('--plot_percentage', type=int, default=15)
    parser.add_argument('--out_dir', default='/tmp/tsne_out')
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args(argv[1:])

    prng = np.random.RandomState(seed=42)

    class_colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow',
                    'black']

    class_dirs = [d for d in os.listdir(args.image_dir) if os.path.isdir(
            os.path.join(args.image_dir, d))]

    if len(class_dirs) > len(class_colors):
        raise ValueError('too many classes for plot')

    class2color = {c : class_colors[i % len(class_colors)] \
            for i, c in enumerate(class_dirs)}

    legend_patches = [mpatches.Patch(color=class2color[c],
                                     label=c) for c in class_dirs]

    data = [(
            c,
            os.path.join(args.bottleneck_dir, c, f + '.txt'),
            os.path.join(args.image_dir, c, f),
            class2color[c],
            prng.randint(100)
            ) for cc, c in enumerate(class_dirs) \
                    for f in os.listdir(os.path.join(args.image_dir, c)) \
                            if not f.startswith('.') \
                            and os.path.exists(os.path.join(
                                    args.bottleneck_dir, c, f + '.txt'))]

    vecs = np.array([vec_from_string(r[1]) for r in data], dtype=float)

    tsneVecs = TSNE(n_components=2, n_iter=1000, init='pca',
                    random_state=prng).fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(20, 12))

    artists = []
    for r, tsneVec in zip(data, tsneVecs):
        if r[4] < args.plot_percentage:
            imscatter(tsneVec[0], tsneVec[1], r[2], ax=ax, color=r[3])

    ax.update_datalim(np.column_stack([tsneVecs[:, 0], tsneVecs[:, 1]]))
    ax.autoscale()
    ax.legend(handles=legend_patches)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    f_out = open(os.path.join(args.out_dir, 'tsne.png'),
                 'w' if args.force else 'x')

    fig.savefig(f_out)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
