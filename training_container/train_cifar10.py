import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import deepspeed
import copy
import sys

def add_argument(args, extended_parser):

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(extended_parser)

    new_args, _ = parser.parse_known_args()
    new_args.deepspeed_config = args.config_file
    new_args.deepspeed = args.deepspeed_flag
    
    print(f"new args={new_args}")

    return new_args


def parse_initial_args():
    parser = argparse.ArgumentParser(description='CIFAR')

    parser.add_argument('--epochs',
                        default=5,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--config-file',
                        type=str,
                        help='Deepspeed config file'
    )

    parser.add_argument('--deepspeed-flag',                        
                        default=False,
                        action='store_true',
                        help=
                        'Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')
    
    args, unknown = parser.parse_known_args()
    print(f"Pased training args: {args}")
    print(f"Unkonwn args, won't be used: {unknown}")
    
    return args, parser
    

def main():
    
    args, extended_parser = parse_initial_args()
    
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    # .. note::
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # TODO: looks like envs are not propagated to child processes.Hence, SM_CHANNEL_DATA will be underfined.
    data_root= os.environ.get('SM_CHANNEL_DATA','/opt/ml/input/data/data')
    batches_dir= os.path.join(data_root, 'cifar-10-batches-py')
                                   
    trainset = torchvision.datasets.CIFAR10(root=data_root,
                                            train=True,
                                            download=False,
                                            transform=transform
                                           )
    testset = torchvision.datasets.CIFAR10(root=data_root,
                                           train=False,
                                           download=False,
                                           transform=transform)        

        
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    
    
    
    ########################################################################
    # 2. Define a Convolutional Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).

    import torch.nn as nn
    import torch.nn.functional as F


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    args = add_argument(args, extended_parser)

    # Initialize DeepSpeed to use the following features
    # 1) Distributed model
    # 2) Distributed data loader
    # 3) DeepSpeed optimizer
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net, model_parameters=parameters, training_data=trainset)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net.to(device)
    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()

    
    ########################################################################
    # 4. Train the network
    # ^^^^^^^^^^^^^^^^^^^^
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize.

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
                model_engine.local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    ########################################################################
    # 5. Test the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #
    # We have trained the network for 2 passes over the training dataset.
    # But we need to check if the network has learnt anything at all.
    #
    # We will check this by predicting the class label that the neural network
    # outputs, and checking it against the ground-truth. If the prediction is
    # correct, we add the sample to the list of correct predictions.
    #
    # Okay, first step. Let us display an image from the test set to get familiar.

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    ########################################################################
    # Okay, now let us see what the neural network thinks these examples above are:

    outputs = net(images.to(model_engine.local_rank))

    ########################################################################
    # The outputs are energies for the 10 classes.
    # The higher the energy for a class, the more the network
    # thinks that the image is of the particular class.
    # So, let's get the index of the highest energy:
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(model_engine.local_rank))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(
                model_engine.local_rank)).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))

    ########################################################################
    # That looks way better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(model_engine.local_rank))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(model_engine.local_rank)).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == "__main__":
    main()
    
