from os import listdir
import matplotlib.pyplot as plt

import numpy
import re
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.manifold import TSNE

def imscatter(x, y, image, ax=None, zoom=1):
    image_filename = re.search(".*/(.*/.*).txt", image).group(1)
    if ax is None:
        ax = plt.gca()
    image = plt.imread("flower_photos/" + image_filename)
    im = OffsetImage(image, zoom=zoom)
    x, y = numpy.atleast_1d(x, y)
    ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
    ax.add_artist(ab)

files = ["bottleneck/daisy/" + filename for filename in listdir("bottleneck/daisy/")]
files += ["bottleneck/tulips/" + filename for filename in listdir("bottleneck/tulips/")]

vecs = [numpy.fromstring(open(filename, "r").readlines()[0], dtype=float, sep=",") for filename in files]

tsneVecs = TSNE(n_components=2).fit_transform(vecs)

fig, ax = plt.subplots()

artists = []
for file, tsneVec in zip(files, tsneVecs):
    imscatter(tsneVec[0], tsneVec[1], file, zoom=0.1, ax=ax)

ax.update_datalim(numpy.column_stack([tsneVecs[:, 0], tsneVecs[:, 1]]))
ax.autoscale()
plt.show()
