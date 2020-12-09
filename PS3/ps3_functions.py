# ps3_functions.py
# CPSC 453 -- Problem Set 3
#
# This script contains pytorch shells for a Logistic regression model, a feed forward network, and an autoencoder.
#
from torch.nn.functional import softmax
from torch import optim, nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch

class LogisticRegression(nn.Module): # initialize a pytorch neural network module
    def __init__(self): # initialize the model
        super(LogisticRegression, self).__init__() # call for the parent class to initialize
        # you can define variables here that apply to the entire model (e.g. weights, biases, layers...)
        # this model only has two parameters: the weight, and the bias.
        # here's how you can initialize the weight:
        # W = nn.Parameter(torch.zeros(shape)) # this creates a model parameter out of a torch tensor of the specified shape
        # ... torch.zeros is much like numpy.zeros, except optimized for backpropogation. We make it a model parameter and so it will be updated by gradient descent.

        # create a bias variable here
        self.W = nn.Parameter(torch.zeros((784, 10)), requires_grad = True)
        self.b = nn.Parameter(torch.zeros((1, 10)), requires_grad = True)


    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            predictions, a tensor of shape 10. If using CrossEntropyLoss, your model will be trained to put the largest number in the index it believes corresponds to the correct class.
        """
        # put the logic here.
        predictions = torch.matmul(x, self.W) + self.b
        predictions = F.sigmoid(predictions)

        return predictions

class FeedForwardNet(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    # Here, you should place an exact copy of the code from the LogisticRegression class, with two modifications:
    # 1. Add another weight and bias vector to represent the hidden layer
    # 2. In the forward function, add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.
    def __init__(self):
      super(FeedForwardNet, self).__init__()
      self.W1 = nn.Parameter(torch.randn((784, 128)), requires_grad = True)
      self.b1 = nn.Parameter(torch.zeros((1, 128)), requires_grad = True)
      self.W2 = nn.Parameter(torch.randn((128, 10)), requires_grad = True)
      self.b2 = nn.Parameter(torch.zeros((1, 10)), requires_grad = True)

    def forward(self, x):
      predictions = torch.matmul(x, self.W1) + self.b1
      predictions = F.relu(predictions)
      predictions = torch.matmul(predictions, self.W2) + self.b2
      predictions = softmax(predictions)
      return predictions

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.enc_lin1 = nn.Linear(784, 1000)
        self.enc_lin2 = nn.Linear(1000, 500)
        self.enc_lin3 = nn.Linear(500, 250)
        self.enc_lin4 = nn.Linear(250, 2)

        self.dec_lin1 = nn.Linear(2, 250)
        self.dec_lin2 = nn.Linear(250, 500)
        self.dec_lin3 = nn.Linear(500, 1000)
        self.dec_lin4 = nn.Linear(1000, 784)


    def encode(self, x):
        x = F.tanh(self.enc_lin1(x))
        x = F.tanh(self.enc_lin2(x))
        x = F.tanh(self.enc_lin3(x))
        x = self.enc_lin4(x)

        # ... additional layers, plus possible nonlinearities.
        return x

    def decode(self, z):
        # ditto, but in reverse
        z = F.tanh(self.dec_lin1(z))
        z = F.tanh(self.dec_lin2(z))
        z = F.tanh(self.dec_lin3(z))
        z = F.sigmoid(self.dec_lin4(z))

        return z

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)




# initialize the model (adapt this to each model)
model = LogisticRegression()
# initialize the optimizer, and set the learning rate
SGD = torch.optim.SGD(model.parameters(), lr = 5e-2) # This is absurdly high.
# initialize the loss function. You don't want to use this one, so change it accordingly
loss_fn = torch.nn.MultiLabelSoftMarginLoss()
batch_size = 128


def evaluate(model, evaluation_set, loss_fn):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        # TODO: Fill in the rest of the evaluation function.
        losses = []
        sum_total = 0
        for data, targets in evaluation_set:
          data = data.to(device)
          targets = targets.to(device)
          model_input = data.view(-1, 784)
          out = model(model_input)
          arg_maxed = torch.argmax(out, dim = 1)
          
          sum_total += (arg_maxed == targets).float().sum()
          losses.append(loss_fn(out, targets).item())
        loss = sum(losses) / len(losses)
        accuracy =  100 * sum_total / len(evaluation_set.dataset)
    return accuracy, loss

def train(model,loss_fn, optimizer, train_loader, test_loader):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    """
    num_epochs = 150 # obviously, this is too many. I don't know what this author was thinking.
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(num_epochs):
        # loop through each data point in the training set
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            
            # run the model on the data
            model_input = data.view(-1, 784)# TODO: Turn the 28 by 28 image tensors into a 784 dimensional tensor.
            out = model(model_input)

            # Calculate the loss
            loss = loss_fn(out,targets)

            # Find the gradients of our loss via backpropogation
            loss.backward()

            # Adjust accordingly with the optimizer
            optimizer.step()

        # Give status reports every 100 epochs
        if epoch % 100==0:
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            tr_acc, tr_loss = evaluate(model, train_loader, loss_fn)
            te_acc, te_loss = evaluate(model, test_loader, loss_fn)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            test_loss.append(te_loss)
            test_acc.append(te_acc)

            
            print(f" Train accuracy: {tr_acc}. Test accuracy: {te_acc}") #TODO: implement the evaluate function to provide performance statistics during training.

    return train_loss, train_acc, test_loss, test_acc


# ----- Functions for Part 5 -----
def mmd(X,Y, kernel_fn):
    """
    Implementation of Maximum Mean Discrepancy.
    :param X: An n x 1 numpy vector containing the samples from distribution 1.
    :param Y: An n x 1 numpy vector containing the samples from distribution 2.
    :param kernel_fn: supply the kernel function to use.
    :return: the maximum mean discrepancy:
    MMD(X,Y) = Expected value of k(X,X) + Expected value of k(Y,Y) - Expected value of k(X,Y)
    where k is a kernel function
    """

    return mmd


def kernel(A, B):
    """
    A gaussian kernel on two arrays.
    :param A: An n x d numpy matrix containing the samples from distribution 1
    :param B: An n x d numpy matrix containing the samples from distribution 2.
    :return K:  An n x n numpy matrix k, in which k_{i,j} = e^{-||A_i - B_j||^2/(2*sigma^2)}
    """

    return K


import matplotlib.pyplot as plt

def evaluate_ae(model, evaluation_set, loss_fn):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        # TODO: Fill in the rest of the evaluation function.
        losses = []
        sum_total = 0
        for ind, (data, targets) in enumerate(evaluation_set):
          data = data.to(device)
          model_input = data.view(-1, 784)
          out = model(model_input)
          # if ind == 0:
          #   visualise_output(model_input, model)
          sum_total += (out == model_input).float().sum()
          losses.append(loss_fn(out, model_input).item())
        loss = sum(losses) / len(losses)
        accuracy =  100 * sum_total / len(evaluation_set.dataset)
    return accuracy, loss

def train_ae(model,loss_fn, optimizer, train_loader, test_loader):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    """
    num_epochs = 500 # obviously, this is too many. I don't know what this author was thinking.
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    to_graph = {}
    while True:
      for data, targets in test_loader:
        for d, t in zip(data, targets):
          if int(t) not in to_graph.keys():
            to_graph[int(t)] = d
          if len(to_graph) == 10:
            break
        if len(to_graph) == 10:
            break
      if len(to_graph) == 10:
            break

    for epoch in range(num_epochs):
        # loop through each data point in the training set
        for data, targets in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            # run the model on the data
            model_input = data.view(-1, 784)# TODO: Turn the 28 by 28 image tensors into a 784 dimensional tensor.
            out = model(model_input)

            # Calculate the loss
            loss = loss_fn(out, model_input)

            # Find the gradients of our loss via backpropogation
            loss.backward()

            # Adjust accordingly with the optimizer
            optimizer.step()

        # Give status reports every 100 epochs
        if epoch % 100==0:
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            tr_acc, tr_loss = evaluate_ae(model, train_loader, loss_fn)
            te_acc, te_loss = evaluate_ae(model, test_loader, loss_fn)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            test_loss.append(te_loss)
            test_acc.append(te_acc)
            

            
            # im = next(iter(train_loader))[0][0]
            # plt.imshow(im.squeeze())
            # plt.show()
            # # import pdb; pdb.set_trace()

            # with torch.no_grad():
            #   plt.imshow(torch.reshape(model(im.view(-1, 784)), (28,28)).numpy())
            #   plt.show()

            
            print(f" Train Loss: {tr_loss}. Test Loss: {te_loss}") #TODO: implement the evaluate function to provide performance statistics during training.

    for label, data in to_graph.items():
      plt.imshow(data.squeeze())
      plt.show()
      with torch.no_grad():
        plt.imshow(torch.reshape(model(data.view(-1, 784)), (28,28)).numpy())
        plt.show()
    return train_loss, train_acc, test_loss, test_acc

class Ret_Autoencoder(Autoencoder):
  def __init__(self, input_size):
    super(Autoencoder, self).__init__()
    self.enc_lin1 = nn.Linear(input_size, 1000)
    self.enc_lin2 = nn.Linear(1000, 500)
    self.enc_lin3 = nn.Linear(500, 250)
    self.enc_lin4 = nn.Linear(250, 2)

    self.dec_lin1 = nn.Linear(2, 250)
    self.dec_lin2 = nn.Linear(250, 500)
    self.dec_lin3 = nn.Linear(500, 1000)
    self.dec_lin4 = nn.Linear(1000, input_size)

  def decode(self, z):
    # ditto, but in reverse
    z = F.tanh(self.dec_lin1(z))
    z = F.tanh(self.dec_lin2(z))
    z = F.tanh(self.dec_lin3(z))
    z = self.dec_lin4(z)

    return z

def evaluate_ret_ae(model, evaluation_set, loss_fn):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        # TODO: Fill in the rest of the evaluation function.
        losses = []
        sum_total = 0
        for ind, (data, targets) in enumerate(evaluation_set):
          data = data.to(device)
          model_input = data
          out = model(model_input)
          # if ind == 0:
          #   visualise_output(model_input, model)
          sum_total += (out == model_input).float().sum()
          losses.append(loss_fn(out, model_input).item())
        loss = sum(losses) / len(losses)
        accuracy =  100 * sum_total / len(evaluation_set.dataset)
    return accuracy, loss

def train_ret_ae(model,loss_fn, optimizer, train_loader, test_loader):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    """
    num_epochs = 500 # obviously, this is too many. I don't know what this author was thinking.
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(num_epochs):
        # loop through each data point in the training set
        for data, targets in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            # run the model on the data
            model_input = data
            out = model(model_input)

            # Calculate the loss
            loss = loss_fn(out, model_input)

            # Find the gradients of our loss via backpropogation
            loss.backward()

            # Adjust accordingly with the optimizer
            optimizer.step()

        # Give status reports every 100 epochs
        if epoch % 100==0:
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            tr_acc, tr_loss = evaluate_ret_ae(model, train_loader, loss_fn)
            te_acc, te_loss = evaluate_ret_ae(model, test_loader, loss_fn)
            train_loss.append(tr_loss)
            train_acc.append(tr_acc)
            test_loss.append(te_loss)
            test_acc.append(te_acc)
            
            print(f" Train Loss: {tr_loss}. Test Loss: {te_loss}") #TODO: implement the evaluate function to provide performance statistics during training.

    return train_loss, train_acc, test_loss, test_acc