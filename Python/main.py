import sub_mod as sm
import numpy as np
import matplotlib.pyplot as plt

"""
w = weights, b = bias, i = input, h = hidden, o = output, l = label
-> w_i_h = weights from input to hidden layer

number of neurons in hidden layer = 20
number of neurons in output layer = 10
"""


# get data from the database
images, labels = sm.get_mnist()

"""
Remark: a connection between two neurons, e.g., input layer neuron to hidden layer neuron, is a linear function,
the weight (w) defines the slope of the function and the values of bias neuron shifts the function up and down.
f(x) = weight * m + bias 
"""

# Matrix: lower limit, upper limit, number of neurons in h, number of neurons in i - 28 x 28 = 784 pixels
w_i_h = np.random.uniform(-0.5, 0.5, [20, 784])
# Matrix: lower limit, upper limit, number of neurons in o, number of neurons in h
w_h_o = np.random.uniform(-0.5, 0.5, [10, 20])
# colum vectors
b_i_h = np.zeros([20, 1])
b_h_o = np.zeros([10, 1])

learn_rate = 0.01
nr_correct = 0
epochs = 3

for epoch in range(epochs):
    # this loop goes through the entire matrices of images and labels and consequently, img / l are row vectors
    for img, l in zip(images, labels):

        # change vectors to matrices, the @ operator cannot work with vectors and matrices, it uses matrices only
        # there for we change, e.g., img[784,] vector to a img[784,1] matrix
        # Example A = 10; A += 5 -> A = 15
        img.shape += (1,)
        l.shape += (1,)

        # Forward propagation input layer to hidden layer
        # a @ b is matrix multiplication different to np.dot for higher dimensional arrays
        # ColumnVector[20,1] = ColumnVector[20,1] + Matrix[20, 784] @ Matrix[784,1]
        # Requirement: no column of matrix A = no rows of matrix B -> A[20, columns = 784] @ B[rows = 784, 1]
        h_pre = b_i_h + w_i_h @ img
        # Sigmoid function to normalize output between 0 and 1
        h = 1 / (1 + np.exp(-h_pre))
        # Forward propagation hidden layer to output layer
        o_pre = b_h_o + w_h_o @ h
        # Sigmoid function to normalize output between 0 and 1
        o = 1 / (1 + np.exp(-o_pre))

        # Cost / Error function
        # mean squared-error-function
        # 1 / (no output neurons) * sum((output - label value)**2)
        e = 1/len(o) * np.sum((o - l) ** 2, axis=0)
        # is the input is classified correctly?
        # Which output neuron has the highest value? Which label has the highest value?
        # increase counter by 1 if correct else 0 is added
        # this gives a global statement how many files were classified correctly
        # but, you cannot find out which one was not classified correctly afterwards
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # Backpropagation output layer to hidden layer (cost function derivative)
        delta_o = o - l
        # Matrix [10, 20] += - scalar * matrix[10, 1] @ (Matrix[1, 20] = np.transpose(Matrix[20, 1]))
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backpropagation hidden layer to input layer (activation function derivative / derivative of the
        # Sigmoid function -> h)
        # Matrix[20, 10] = matrix[20, 10] @ matrix[10, 1] * matrix[20, 1]
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(img)
        b_i_h += -learn_rate * delta_h

    # Show accuracy (in percentage) for this epoch
    # Remark: print(f".. {replacement by anything}") f indicates an f string;
    # here, a placeholder gets replaced by any given value
    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# Show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)

    # Forward propagation input layer to hidden layer
    h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))

    # Forward propagation hidden layer to output layer
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Subscribe if its a {o.argmax()} :)")
    plt.show()
