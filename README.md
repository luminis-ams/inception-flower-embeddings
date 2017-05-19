# Inception flower embeddings

**Use the Inception image classification model to create embeddings of flower images. Convert these embeddings to a 2d space using TSNe and plot them with matplotlib.**

1. Download the flower images at: http://download.tensorflow.org/example_images/flower_photos.tgz
2. Extract the bottleneck.tar.gz
3. Run tsne_embedding_bottlenecks.py


If you want to create your own embeddings, follow the tutorial at: https://www.tensorflow.org/tutorials/image_retraining (don't forget to run .configure before compiling with bazel). Run the retrain script with your own images and use that image and bottleneck dir in the tsne_embedding_bottlenecks.py script

#### Examples

Daisy vs Tulips:

![Daisy vs Tulips](examples/daisy_vs_tulips.png?raw=true "Daisy vs Tulips")

Roses vs Sunflowers:

![Roses vs Sunflowers](examples/roses-vs-sunflowers.png?raw=true "Roses vs Sunflowers")