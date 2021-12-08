import numpy as np
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-9  # add small epsilon for numerical stability


# read and normalize the data


class MinMaxScaler():
    """
    A Min-Max Scaler that is to be fit to the train data.
    """
    def __init__(self, x_train):
        self.x_min = np.min(x_train, axis=0)
        self.x_max = np.max(x_train, axis=0)

    def scale(self, x):
        """
        A method used to scale new data based on the MinMaxScaler that was defined by the train data.
        :param x: Data matrix to be scaled
        :return: Scaled data
        """
        x_scaled = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
        return x_scaled


def data_read(train_val_ratio):
    train_data = pd.read_csv('train.csv').to_numpy()
    test_data = pd.read_csv('test.csv').to_numpy()

    N = train_data.shape[0]
    indices = np.random.permutation(N)
    indices_train, indices_val = indices[:int(N * train_val_ratio)], indices[int(N * train_val_ratio):]

    x_train = train_data[indices_train, 1:]
    y_train = train_data[indices_train, 0]
    x_val = train_data[indices_val, 1:]
    y_val = train_data[indices_val, 0]
    x_test = test_data

    y_train_one_hot = one_hot_encode(y_train).T
    y_val_one_hot = one_hot_encode(y_val).T

    scaler = MinMaxScaler(x_train)
    x_train = scaler.scale(x_train).T
    x_val = scaler.scale(x_val).T
    x_test = scaler.scale(x_test).T
    return x_train, y_train_one_hot, x_val, y_val_one_hot, x_test


def one_hot_encode(y):
    y_one_hot = np.zeros((y.size,10))
    y_one_hot[np.arange(y.size),y] = 1
    return y_one_hot


def visualize_data(x_train, y_train):
    fig, axes = plt.subplots(10, 4, figsize=(12, 5))
    labels = np.argmax(y_train, axis=0)
    for label in range(10):
        indices_of_label = np.where(labels == label)[0]
        for i in range(4):
            if i == 0:
                axes[label, i].set_ylabel('class '+str(label),verticalalignment='center')

            axes[label, i].imshow(x_train[:, indices_of_label[i]].reshape(28, 28), cmap='Greys')
            axes[label, i].set_xticks([])
            axes[label, i].set_yticks([])
    return fig


# logistic regression

def test(x, w, l2_lambda, y_true=None):
    z = np.matmul(w, x)
    softmax_z = softmax(z)
    predictions = np.argmax(softmax_z, axis=0)
    if y_true is None:
        return predictions

    else:
        test_loss = cross_entropy_loss(softmax_z, y_true) + l2_reg(w, l2_lambda)
        accuracy = np.mean(np.equal(np.argmax(y_true, axis=0), predictions))
        return accuracy, test_loss


def logistic_regression(x_train, y_train, x_val, y_val, x_test, batch_size, lr, epochs, l2_lambda):
    # add a feature for the bias

    x_train_full = np.vstack([x_train, np.ones([1, x_train.shape[1]])])
    x_val_full = np.vstack([x_val, np.ones([1, x_val.shape[1]])])
    x_test_full = np.vstack([x_test, np.ones([1, x_test.shape[1]])])

    N = x_train_full.shape[1]
    w_dim = x_train_full.shape[0]

    w = np.random.normal(loc=0.0, scale=0.01,
                         size=(10, w_dim))  # initialize weights iid from a gaussian with small noise
    train_losses, train_accuracy, val_losses, val_accuracy = [], [], [], []

    # iterations over entire dataset
    for epoch in range(epochs):
        loss = 0
        # batch iterations within each dataset iteration
        for batch_idx, idx_start in enumerate(range(0, N, batch_size)):
            idx_end = min(idx_start + batch_size, N)
            x_batch = x_train_full[:, idx_start:idx_end]  # take all data in the current batch
            y_batch = y_train[:, idx_start:idx_end]  # .reshape(-1, 1)  # take relevant labels
            # matrix-vector multiplication
            z = np.matmul(w, x_batch)
            # calc. probability of y_j = 1 for each input (batch_size,)
            softmax_z = softmax(z)

            # calculate loss

            batch_loss = cross_entropy_loss(softmax_z, y_batch) + l2_reg(w,l2_lambda)
            loss += batch_loss
            # compute gradient of the loss w.r.t w

            delta_w = dL_dw(softmax_z, y_batch, x_batch) + 2 * l2_lambda * w
            # update w
            w = w - lr * delta_w

        # validation
        val_acc, val_loss = test(x_val_full, w, l2_lambda, y_true=y_val)
        train_acc, _ = test(x_train_full, w, l2_lambda, y_true=y_train)
        # save for plotting
        train_losses.append(loss / (len(range(0, N, batch_size)) - 1))
        train_accuracy.append(train_acc)

        val_losses.append(val_loss)
        val_accuracy.append(val_acc)
        print('Epoch:' + str(epoch) + '    Train Acc: ' + str(train_acc) + '    Val Acc: ' + str(val_acc))

    # save the test results to a file
    test_preds = test(x_test_full, w, l2_lambda, y_true=None).astype(int)
    file_name = 'lr_pred.csv'
    print(test_preds)
    # np.savetxt(file_name, test_preds, fmt='%i')

    return train_losses, train_accuracy, val_losses, val_accuracy


# neural network

def test_NN(x, w1, b1, w2, b2, l2_lambda, y=None):
    z1 = np.matmul(w1, x) + b1
    # h = sigmoid(z1)
    h = relu(z1)
    z2 = np.matmul(w2, h) + b2
    y_pred = np.exp(z2) / (np.sum(np.exp(z2), axis=0) + epsilon)
    preds = np.argmax(y_pred, axis=0)
    if y is None:  # test case
        return preds
    else:  # val case
        # calculate loss
        regularization_term = l2_lambda * (np.sum(np.power(w1, 2)) + np.sum(np.power(w2, 2)))
        test_loss = cross_entropy_loss(y_pred, y) + regularization_term
        # calc. accuracy
        accuracy = np.mean(np.equal(np.argmax(y, axis=0), preds))
        return accuracy, test_loss


class NN():
    def __init__(self, NN_width, input_dim=784, act_type='relu'):

        self.input_dim = input_dim
        self.w1 = 0.1*np.random.randn(NN_width, self.input_dim)
        # self.w1 = np.random.randn(NN_width, self.input_dim)
        self.b1 = np.zeros((NN_width, 1))
        self.w2 = 0.1*np.random.randn(10, NN_width)
        # self.w2 = np.random.randn(10, NN_width)
        self.b2 = np.zeros((10, 1))
        self.epoch = 0
        self.activation = relu if act_type == 'relu' else (sigmoid if act_type == 'sigmoid' else 'Error')
        if self.activation == 'Error':
            raise IOError('activation type could be either \'relu\' or \'sigmoid\'')

        self.train_losses = []
        self.train_accuracy = []
        self.val_losses = []
        self.val_accuracy = []

    def train(self, x_train, y_train, x_val, y_val, batch_size, lr, num_epochs, l2_lambda=0, dropout_keep_prob = 1):
        train_samples = x_train.shape[1]  # number of training samples

        # iterations over entire dataset
        for epoch in range(num_epochs):
            self.epoch += 1
            loss = 0
            # batch iterations within each dataset iteration
            for batch_idx, idx_start in enumerate(range(0, train_samples, batch_size)):
                idx_end = min(idx_start + batch_size, train_samples)
                x = x_train[:, idx_start:idx_end]  # take all data in the current batch
                y = y_train[:, idx_start:idx_end]  # .reshape(-1, 1)  # take relevant labels

                z1 = np.matmul(self.w1, x) + self.b1
                h = self.activation(z1)  # sigmoid / relu
                #dropout - scaled down so there is no need to change test code
                dropout_mask_1 = (np.random.rand(*h.shape) < dropout_keep_prob) / dropout_keep_prob 
                h *= dropout_mask_1
                z2 = np.matmul(self.w2, h) + self.b2
                y_pred = np.exp(z2) / (np.sum(np.exp(z2), axis=0))
                loss += cross_entropy_loss(y_pred, y) + l2_reg(self.w1, l2_lambda) + l2_reg(self.w2, l2_lambda)

                # compute gradient of the loss
                dLdz2 = y_pred - y
                dLdb2 = (1. / train_samples) * np.sum(dLdz2, axis=1, keepdims=True)
                dLdw2 = (1. / train_samples) * np.matmul(dLdz2, h.T)

                dLdh = np.matmul(self.w2.T, dLdz2)
                if self.activation == relu:
                    dLdz1 = dLdh * relu_derivative(z1)
                else:
                    dLdz1 = dLdh * sigmoid(z1) * (1 - sigmoid(z1))
                # if dopout applied in forward prop - bakprop shuld mask and scale too
                dLdh *= dropout_mask_1
                dLdw1 = (1. / train_samples) * np.matmul(dLdz1, x.T)
                dLdb1 = (1. / train_samples) * np.sum(dLdz1, axis=1, keepdims=True)

                self.w2 = self.w2 - lr * (dLdw2 + 2 * l2_lambda * self.w2)
                self.b2 = self.b2 - lr * dLdb2
                self.w1 = self.w1 - lr * (dLdw1 + 2 * l2_lambda * self.w1)
                self.b1 = self.b1 - lr * dLdb1

            train_acc, train_loss = self.test(x_train, l2_lambda, y=y_train)
            val_acc, val_loss = self.test(x_val, l2_lambda, y=y_val)
            print('Epoch:' + str(self.epoch) + '    Train Acc: ' + str(train_acc) + '    Val Acc: ' + str(val_acc))

            # save for plotting
            self.train_losses.append(loss / (len(range(0, train_samples, batch_size)) - 1))
            self.train_accuracy.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracy.append(val_acc)

        return self.train_losses, self.train_accuracy, self.val_losses, self.val_accuracy

    def test(self, x, l2_lambda, y=None):
        z1 = np.matmul(self.w1, x) + self.b1
        h = relu(z1)
        z2 = np.matmul(self.w2, h) + self.b2
        y_pred = np.exp(z2) / (np.sum(np.exp(z2), axis=0) + epsilon)
        predictions = np.argmax(y_pred, axis=0)
        if y is None:  # test case
            return predictions
        else:  # val case
            # calculate loss
            test_loss = cross_entropy_loss(y_pred, y) + l2_reg(self.w1, l2_lambda) + l2_reg(self.w2, l2_lambda)
            # calc. accuracy
            accuracy = np.mean(np.equal(np.argmax(y, axis=0), predictions))
            return accuracy, test_loss

    def save_test_predictions(self):
        test_predictions = self.test(x_test, l2_lambda, y=None).astype(int)
        file_name = 'NN_pred.csv'
        np.savetxt(file_name, test_predictions, fmt='%i')  # d = 500, Bsize = 1024, LR=1, epoch=50 acc = 0.83
        return

def neural_net(x_train, y_train, x_val, y_val, x_test, batch_size, lr, num_epochs, l2_lambda, NN_width, dropout_keep_prob = 1):
    train_samples = x_train.shape[1]  # number of training samples
    input_dim = x_train.shape[0]  # dimension 784

    w1 = np.random.randn(NN_width, input_dim)
    b1 = np.zeros((NN_width, 1))
    w2 = np.random.randn(10, NN_width)
    b2 = np.zeros((10, 1))

    train_losses, train_accuracy, val_losses, val_accuracy = [],[],[],[]

    # iterations over entire dataset
    for epoch in range(num_epochs):
        loss = 0
        accuracy = 0
        print(epoch)
        # batch iterations within each dataset iteration
        for batch_idx, idx_start in enumerate(range(0, train_samples, batch_size)):
            idx_end = min(idx_start + batch_size, train_samples)
            x = x_train[:, idx_start:idx_end]  # take all data in the current batch
            y = y_train[:, idx_start:idx_end]  # .reshape(-1, 1)  # take relevant labels
            
            # 1st layer
            z1 = np.matmul(w1, x) + b1
            # h = sigmoid(z1)
            h = relu(z1)  # sigmoid / relu
            #dropout - scaled down so there is no need to change test code
            dropout_mask_1 = (np.random.rand(*h.shape) < dropout_keep_prob) / dropout_keep_prob 
            h *= dropout_mask_1

            # 2nd layer
            z2 = np.matmul(w2, h) + b2
            y_pred = np.exp(z2) / np.sum(np.exp(z2), axis=0)
            regularization_term = l2_lambda * (np.sum(np.power(w1, 2)) + np.sum(np.power(w2, 2)))
            loss += cross_entropy_loss(y_pred, y) + regularization_term
            preds = np.argmax(y_pred, axis=0)
            batch_accuracy = np.mean(np.equal(np.argmax(y, axis=0), preds))
            accuracy += batch_accuracy

            # compute gradient of the loss
            dLdz2 = y_pred - y
            dLdb2 = (1. / train_samples) * np.sum(dLdz2, axis=1, keepdims=True)
            dLdw2 = (1. / train_samples) * np.matmul(dLdz2, h.T)

            dLdh = np.matmul(w2.T, dLdz2) 
            # if dopout applied in forward prop - bakprop shuld mask and scale too
            dLdh *= dropout_mask_1
            # dLdz1 = dLdh * sigmoid(z1) * (1 - sigmoid(z1))
            dLdz1 = dLdh * relu_derivative(z1)
            dLdw1 = (1. / train_samples) * np.matmul(dLdz1, x.T)
            dLdb1 = (1. / train_samples) * np.sum(dLdz1, axis=1, keepdims=True)

            # update
            regularization_term_w1 = 2 * l2_lambda * w1
            regularization_term_w2 = 2 * l2_lambda * w2

            w2 = w2 - lr * (dLdw2 + regularization_term_w2)
            b2 = b2 - lr * dLdb2
            w1 = w1 - lr * (dLdw1 + regularization_term_w1)
            b1 = b1 - lr * dLdb1

        train_acc, train_loss = test_NN(x_train, w1, b1, w2, b2, l2_lambda, y=y_train)
        print('epoch: ' + str(epoch) + '\ntrain_acc: ' + str(train_acc))

        # validation
        val_acc, val_loss = test_NN(x_val, w1, b1, w2, b2, l2_lambda, y=y_val)

        # save for plotting
        train_losses.append(loss / (len(range(0, train_samples, batch_size)) - 1))
        train_accuracy.append(accuracy / (len(range(0, train_samples, batch_size)) - 1))
        # train_losses.append(train_loss)
        # train_accuracy.append(train_acc)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)

    # save the test results to a file
    test_preds = test_NN(x_test, w1, b1, w2, b2, l2_lambda, y=None).astype(int)
    file_name = 'NN_pred.csv'
    np.savetxt(file_name, test_preds, fmt='%i')  # d = 500, Bsize = 1024, LR=1, epoch=50 acc = 0.83

    return train_losses, train_accuracy, val_losses, val_accuracy


# activation functions

def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    y = np.zeros_like(x)
    y[x > 0] = 1
    return y


def softmax(a):
    d = a - np.matmul(np.ones((a.shape[0], 1)) , (np.max(a, axis=0).reshape(1, a.shape[1])))
    return np.exp(d) / np.sum(np.exp(d), axis=0).reshape(1, d.shape[1])

# machine learning functions

def cross_entropy_loss(softmax_z, y_true_one_hot):
    loss = np.sum(np.log(softmax_z + epsilon) * y_true_one_hot)
    return - loss / softmax_z.shape[1]


def l2_reg(w, l2_lambda):
    reg_term = l2_lambda * np.sum(np.power(w,2))
    return reg_term


def dL_dw(softmax_z, y_true, x):
    dL = np.matmul((y_true - softmax_z), x.T)
    return - dL / softmax_z.shape[1]


# plot the loss and accuracy

def show_learning_curve(train_loss_list, val_loss_list, train_accuracy, val_accuracy, num_epochs,
                        batch_size, lr, l2_lambda, dropout_keep_prob=1, NN_width=0):
    fig, axes = plt.subplots(1, 2)

    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('loss')
    axes[0].plot(range(num_epochs), train_loss_list, label="Train", color='blue')
    axes[0].plot(range(num_epochs), val_loss_list, label="Validation", color='red')
    axes[0].legend()

    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel('accuracy')  # we already handled the x-label with ax1
    axes[1].plot(range(num_epochs), train_accuracy, label="Train", color='blue')
    axes[1].plot(range(num_epochs), val_accuracy, label="Validation", color='red')
    axes[1].legend()

    if NN_width == 0:
        fig.suptitle('Logistic Regression:\n\n learning rate = {}   |   batch size = {}   |   L2 lambda = {}'
                     .format(lr, batch_size, l2_lambda))
    else:
        fig.suptitle('Neural Network:\n\n learning rate = {}  |   batch size = {}   |   L2 lambda = {}'
                     '  |   NN width = {}    |   dropout (keep prob) = {}'.format(lr, batch_size, l2_lambda, NN_width, dropout_keep_prob))
    return fig

print('checkpoint')

#####################################
# Part 1 - Read Data and Visualize
#####################################
train_val_ratio = 0.1
# read the data
x_train, y_train, x_val, y_val, x_test = data_read(train_val_ratio)
visualize_data(x_train, y_train)



#####################################
# Part 2 - Logistic Regression
#####################################
# hyper-parameters
num_epochs = 500
batch_size = 1024
lr = 1
lr = 0.1
l2_lambda = 0
NN_width = 500

train_loss_list, train_accuracy, val_loss_list, val_accuracy =\
    logistic_regression(x_train, y_train, x_val, y_val, x_test, batch_size, lr, num_epochs, l2_lambda)

show_learning_curve(train_loss_list, val_loss_list, train_accuracy, val_accuracy, num_epochs, batch_size, lr, l2_lambda)



#####################################
# Part 3 - Neural Network
#####################################
# hyper-parameters
num_epochs = 30  # number of times to iterate over the entire dataset
batch_size = 1024  # batch size
lr = 1
l2_lambda = 0.00001
NN_width = 500  # NN layer size
dropout_keep_prob=0.5



nn = NN(NN_width, act_type='relu')
train_loss_list, train_accuracy, val_loss_list, val_accuracy = nn.train(x_train, y_train, x_val, y_val, batch_size, lr, num_epochs, l2_lambda, dropout_keep_prob=dropout_keep_prob )
show_learning_curve(train_loss_list, val_loss_list, train_accuracy, val_accuracy, nn.epoch, batch_size, lr, l2_lambda, NN_width=NN_width, dropout_keep_prob=dropout_keep_prob)
plt.show()
nn.save_test_predictions()