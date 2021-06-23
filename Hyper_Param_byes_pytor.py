import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F3
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images

dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


def net_train(net, train_loader, parameters, dtype, device):
    net.to(dtype=dtype, device=device)
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), # or any optimizer you prefer 
                            lr=parameters.get("lr", 0.001), # 0.001 is used if no lr is specified
                            momentum=parameters.get("momentum", 0.9)
                        )
    scheduler = optim.lr_scheduler.StepLR(
                                        optimizer,
                                        step_size=int(parameters.get("step_size", 30)),
                                        gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
                                        )
    num_epochs = parameters.get("num_epochs", 3) # Play around with epoch number
    # Train Network
    for _ in range(num_epochs):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net

def init_net(parameterization):
    model = torchvision.models.resnet50(pretrained=True) #pretrained ResNet50
    # The depth of unfreezing is also a hyperparameter
    for param in model.parameters():
        param.requires_grad = False # Freeze feature extractor
    Hs = 512 # Hidden layer size; you can optimize this as well

    model.fc = nn.Sequential(nn.Linear(2048, Hs), # attach trainable classifier
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(Hs, 10),
                                 nn.LogSoftmax(dim=1))
    return model # return untrained model


# 最后，我们需要一个train_evaluate()函数，该函数在每次运行时都会被贝叶斯优化器调用。 
# 优化器在parameterization化中生成一组新的超parameterization ，将其传递给此函数，
# 然后分析返回的评估结果。
def train_evaluate(parameterization):
     
    # constructing a new training data loader allows us to tune the batch size
    train_loader = torch.utils.data.DataLoader(trainset,
                                batch_size=parameterization.get("batchsize", 32),
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)
    # Get neural net
    untrained_net = init_net(parameterization)
    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader, 
                            parameters=parameterization, dtype=dtype, device=device)
    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=testloader,
        dtype=dtype,
        device=device,
    )


# 现在，只需指定要扫描的超参数并将其传递给Ax的optimize()函数即可
#torch.cuda.set_device(0) #this is sometimes necessary for me
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "batchsize", "type": "range", "bounds": [16, 128]},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        #{"name": "max_epoch", "type": "range", "bounds": [1, 30]},
        #{"name": "stepsize", "type": "range", "bounds": [20, 40]},        
    ],
  
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)

print(best_parameters)
means, covariances = values
print(means)
print(covariances)