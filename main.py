import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
from resnet import ResNet18



model_num = 4 # total number of models
total_epoch = 100 # total epoch
lr = 0.1 # initial learning rate

mixup_alpha = 1.0

 # Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mixup_data(x, y):
    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, y, targets, smoothing=0.1):
        confidence = 1. - smoothing
        log_probs = F.log_softmax(y, dim=-1) # 예측 확률 계산
        true_probs = torch.zeros_like(log_probs)
        true_probs.fill_(smoothing / (y.shape[1] - 1))
        true_probs.scatter_(1, targets.data.unsqueeze(1), confidence) # 정답 인덱스의 정답 확률을 confidence로 변경
        return torch.mean(torch.sum(true_probs * -log_probs, dim=-1)) # negative log likelihood



for s in range(model_num):

    seed_number = s
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Define the data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,4),
        transforms.AutoAugment(),  
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    # Load the CIFAR-10 dataset

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,num_workers=12,pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=12,pin_memory=True)

    # Define the ResNet-18 model with pre-trained weights

    #model = timm.create_model('resnet18', pretrained=False, num_classes=10)
    model = ResNet18()
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    #criterion = LabelSmoothingCrossEntropy()
    #optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.99,0.999))
    optimizer = optim.SGD(model.parameters(),lr,momentum=0.9,weight_decay=0.0002)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    def train():
        model.train()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            inputs,targets_a,targets_b,lam = mixup_data(inputs,labels)

            #optimizer.zero_grad()

            for param in model.parameters():
                param.grad = None

            outputs = model(inputs)

            #loss = criterion(outputs, labels)
            loss = mixup_criterion(criterion,outputs,targets_a,targets_b,lam)
            loss.backward()

            optimizer.step()

            running_loss += loss.detach()

            nn.utils.clip_grad_value_(model.parameters(), 0.1)

            if i % 100 == 99:

                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / inputs.size(0)))

                running_loss = 0.0   

                

    def test():

        model.eval()

        # Test the model

        correct = 0
        total = 0

        with torch.no_grad():

            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                #images,targets_a,targets_b,lam = mixup_data(images,labels)

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0) 
                correct += (predicted == labels).sum().detach()


        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0,1,2,3]) 
        model.to(device=device)


    # Train the model

    for epoch in range(total_epoch):

        train()
        test()

        scheduler.step()




    print('Finished Training')


    # Save the checkpoint of the last model

    PATH = '/workspace/resnet/new_resnet18_cifar10_%f_%d.pth' % (lr, seed_number)

    torch.save(model.state_dict(), PATH)
