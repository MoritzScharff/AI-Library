import numpy as np
import pathlib


def get_mnist():
    """

    :return: images and labels as ndarrays
    """
    # the with statement replaces open / read / close
    # the data set gets imported and is called f
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]

    images = images.astype("float32") / 255
    # reshape(data, new shape) -> from 3D [60000, 28, 28] to 2D [60000, 28*28=784]
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    # reshape from [60000,1] to [60000, 10], labels in binary format / one-hot encoded
    # there are ten possible outputs / labels 0..9 so there are 10 out put neurons
    # the labels are used to check the value of the output neuron, e.g., if the image is a 3,
    # output neuron four should be one and the rest zero. In the beginning, the output neurons will show any values
    # and therefore, we need to check the output compared to the label of the dataset
    labels = np.eye(10)[labels]

    return images, labels
