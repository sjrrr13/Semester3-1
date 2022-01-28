import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import default_loader

EPOCH = 50
BATCH_SIZE = 114
LR = 0.0005
DECAY = 0.0001

train_root = "./train/"
test_root = "./test/"

data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

test_data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(train_root, transform=data_transform,
                                              target_transform=None, loader=default_loader, is_valid_file=None)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.ImageFolder(test_root, transform=test_data_transform,
                                             target_transform=None, loader=default_loader, is_valid_file=None)
test_loader = data.DataLoader(dataset=test_data, batch_size=12 * 50 - 1, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())

        self.fc1 = nn.Linear(7 * 7 * 32, 24)
        self.fc2 = nn.Linear(24, 12)
        # self.fc = nn.Linear(7 * 7 * 16, 12)
        self.result = nn.Softmax(dim=0)

    # forward
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        # out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.result(out)
        return out


if __name__ == "__main__":
    save_path = "/Users/sjrrr/Fudan/Semester3-1/IntelligentSystem/Labs/Lab1Project/CNNModel/cnn_5.pth"
    network = torch.load(save_path)
    # network = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)

    for epoch in range(EPOCH):
        for batch, (image, label) in enumerate(train_loader):
            output = network(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (image, label) in enumerate(test_loader):
                output = network.forward(image)
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        print('Accuracy: {}%'.format(100 * correct / total))

    torch.save(network, save_path)
