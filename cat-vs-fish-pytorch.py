import torch
import torchvision
from torchvision import transforms

# the transforms method resize all the images to the same size power of 2 to be applicable to the GPU
#then change the images into tensor formates
#then normalize all the inputs to have the same mean and standard deviation

train_data_path = ''
transforms =  transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.255])])
                               
train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform=transforms)

val_data_path = ''
val_data = torchvision.datasets.ImageFolder(root = val_data_path, transform= transforms)

test_data_path = ''
test_data = torchvision.datasets.ImageFolder(root = test_data_path, transform=transforms)


class SimpleNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12288, 84)
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50,2)
    
    def forward(self):
        x = x.view(-1, 12288)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

simplenet = SimpleNet()

if torch.cuda.is_available():
    device = torch.device('cuda')

else:
    device = torch.device('cpu')

simplenet.to(device)

import torch.optim as optim
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

batch_size=64
train_data_loader = data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = data.DataLoader(test_data, batch_size=batch_size)

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs = 20, device = 'cpu'):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            target = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_data_loader)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1],
            target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)
        print('Epoch: {}, Training Loss: {:.2f},Validation Loss: {:.2f},  accuracy = {:.2f}'.format(epoch, training_loss,
                valid_loss, num_correct / num_examples))

train(simplenet, optimizer, torch.nn.CrossEntropyLoss(),train_data_loader, test_data_loader,device)
from PIL import Image
labels = ['cat','fish']
img = Image.open(FILENAME)
img = transforms(img)
img = img.unsqueeze(0)
prediction = simplenet(img)
prediction = prediction.argmax()
print(labels[prediction])