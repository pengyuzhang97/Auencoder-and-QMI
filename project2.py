import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import time
import math
from scipy import stats
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np

import random
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm

import map_classifier

from sklearn import metrics

from sklearn import svm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm :
    - classes :
    - normalize :
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, outputdimen)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

# mlp = Net()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_datasets(batch_size):
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # obtain training indices that will be used for validation
    num_train = len(train_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)

    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)

    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              num_workers=0)

    return train_loader, test_loader, valid_loader


train_set = torchvision.datasets.FashionMNIST("./data", download=False, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=False, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))


#==============================================================================================================
# batch_size = 200
# n_epochs = 1
# lr = 0.01
# sigma = 0.4

# train_loader, test_loader, valid_loader = create_datasets(batch_size)


def train_model(model, batch_size, patience, n_epochs):
    # to track the training loss as the model trains
    train_losses = np.zeros((n_epochs, 240))
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):

        train_loss = 0
        for step, (b_x, b_y) in tqdm(enumerate(train_loader)):

            if use_cuda:
                b_x = b_x.cuda()
                b_y = b_y.cuda()

            output = mlp(b_x.float().to(device))

            VJ = 0
            for k in range(10):
                temp_output = output[np.where(b_y.cpu().numpy() == k)]
                temp_sum_density = 0
                for n1 in range(len(temp_output)):
                    temp_difference = ((temp_output[n1] - temp_output) / sigma) ** 2
                    temp_pro_density = torch.prod(
                        1 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(-temp_difference / 2), 1)
                    temp_sum_density = temp_sum_density + torch.sum(temp_pro_density)
                VJ = VJ + temp_sum_density
            VJ = VJ / (len(b_x) ** 2)

            VM = 0
            for n1 in range(len(output)):
                temp_difference = ((output[n1] - output) / sigma) ** 2
                temp_pro_density = torch.prod(1 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(-temp_difference / 2), 1)
                VM = VM + torch.sum(temp_pro_density)
            VM = VM / (len(b_x) ** 2)
            NT = 0
            for k in range(10):
                NT = NT + (len(output[np.where(b_y.cpu().numpy() == k)]) / len(output)) ** 2
            VM = VM * NT

            VC = 0
            for k in range(10):
                temp_output = output[np.where(b_y.cpu().numpy() == k)]
                temp_sum_density = 0
                for n1 in range(len(temp_output)):
                    temp_difference = ((temp_output[n1] - output) / sigma) ** 2
                    temp_pro_density = torch.prod(
                        1 / (sigma * np.sqrt(2 * np.pi)) * torch.exp(-temp_difference / 2), 1)
                    temp_sum_density = temp_sum_density + torch.sum(temp_pro_density)
                VC = VC + temp_sum_density * len(temp_output) / len(b_x)
            VC = VC / (len(b_x) ** 2)

            loss = -(VJ + VM - 2 * VC)

            # loss = criterion(output, target)
            optimizer.zero_grad()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses[epoch, step] = loss.item()


    return model, train_losses


outputdimen = 20

batch_size = 200
n_epochs = 1
lr = 0.0001
sigma = 0.1


mlp = Net()
use_cuda = 1
if use_cuda:
    mlp = mlp.cuda()

optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []


train_loader, test_loader, valid_loader = create_datasets(batch_size)


model, train_loss = train_model(mlp, batch_size, 1, n_epochs)




mlp.eval()
output_final = np.zeros((48000, outputdimen))
output_finallabel = np.zeros((48000, 1))
with torch.no_grad():
    for step1, (b_x, b_y) in tqdm(enumerate(train_loader)):
        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output1 = mlp(b_x.float().to(device))
        output_final[step1*200:(step1+1)*200, 0:outputdimen] =  output1.cpu().numpy()
        output_finallabel[step1*200:(step1+1)*200, 0] = b_y.cpu().numpy()




output_finallabel = np.squeeze(output_finallabel, 1)

# np.save('traindata after MLP', output_final)
# np.save('traindata_label after MLP', output_finallabel)

clf = svm.SVC()
clf.fit(output_final, output_finallabel)

# clf = map_classifier.MAPClassifier()
# clf.fit(output_final, output_finallabel)


'''test data'''
mlp.eval()
test_final = np.zeros((10000, outputdimen))
test_finallabel = np.zeros((10000, 1))
with torch.no_grad():
    for step1, (b_x, b_y) in tqdm(enumerate(test_loader)):
        if use_cuda:
            b_x = b_x.cuda()
            b_y = b_y.cuda()

        output1_test = mlp(b_x.float().to(device))
        test_final[step1*200:(step1+1)*200, 0:outputdimen] =  output1_test.cpu().numpy()
        test_finallabel[step1*200:(step1+1)*200, 0] = b_y.cpu().numpy()


# np.save('testdata', test_final)
# np.save('testdata_label', test_finallabel)


test_pred = clf.predict(test_final)

# test_pred = clf.predict(test_final)

test_accuracy = metrics.accuracy_score(test_finallabel, test_pred)

print(test_accuracy)

plt.figure(1)
plt.plot(-train_loss[n_epochs-1, :] / np.max(-train_loss[n_epochs-1, :]))
plt.title('Normalized Quadratic Mutual Information')
plt.xlabel('Batch')
plt.ylabel('Normalized by the maximum value')


'''CM of Train data'''
cluster = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
CM = confusion_matrix(test_finallabel, test_pred)
plt.figure(2)
plot_confusion_matrix(CM, cluster, normalize=True, title='CM_Test')
plt.tight_layout()
plt.savefig('./image/diff lr/CM_%.4f.png'% (lr))
plt.show()