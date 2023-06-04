import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm


model_num = 4
lr = 0.01

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Resize(256),
#         # transforms.RandomCrop(224),
#         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#     ])

# Define the data transforms
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(transforms.ToTensor()(crop)) for crop in crops])),
])
# Load the CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=12)

# Define the list of models for ensemble
models = []
for i in range(model_num):
    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', num_classes=10)
    model = nn.DataParallel(model,device_ids=[0,1,2,3]) 
    model.load_state_dict(torch.load(f"/workspace/resnet/resnet18_cifar10_%f_%d.pth" % (lr, i)))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)  # Move the model to the GPU
    models.append(model)

# Evaluate the ensemble of models
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
        bs, ncrops, c, h, w = images.size()       
        outputs = torch.zeros(bs, 10).to(device)  # Initialize the output tensor with zeros
        for model in models:
            model_output = model(images.view(-1, c, h, w))  # Reshape the input to (bs*10, c, h, w)
            model_output = model_output.view(bs, ncrops, -1).mean(1)  # Average the predictions of the 10 crops
            model_output = model(images)
            outputs += model_output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the ensemble on the 10000 test images: %f %%' % (100 * correct / total))