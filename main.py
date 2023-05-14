import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torchvision.models as models
from save_results import save_data


class ResNetModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.resnet = models.resnet50(pretrained=True)
    for param in self.resnet.parameters():
      param.requires_grad = False

    self.resnet.fc = nn.Sequential(
        nn.Linear(2048, 163),
    )
  
  def forward(self, x):
    return self.resnet(x)
  

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.AdaptiveAvgPool2d(output_size=(2,2)),
        nn.Flatten(),
        nn.Linear(256*2*2, 64),
        nn.Linear(64, 163)
    )
  
  def forward(self, x):
    return self.model(x)
  

def train_function(model, train_dataloader, loss_fn, device):
    LR=0.001
    EPOCHS=100
    best_model = None
    best_loss = float('inf')
    train_losses = []
    train_accs = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_total = 0
        train_correct = 0
        model.train()
        for images, labels in train_dataloader:
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(images)
          loss = loss_fn(outputs.squeeze(1), labels)
          loss.backward()
          optimizer.step()

          train_loss += loss.item() * images.size(0)
          train_total += labels.size(0)
          _, predicted = torch.max(outputs.data, 1)
          train_correct += (predicted == labels).sum().item()

        epoch_loss = train_loss / len(train_dataloader.dataset)
        if best_loss > epoch_loss:
           best_loss = epoch_loss
           best_model = model.state_dict()

        train_losses.append(epoch_loss)
        train_accs.append(train_correct / train_total)
        
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch+1, EPOCHS, train_losses[-1], train_accs[-1]))

    return best_model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train = ImageFolder(root='./train', transform=transform)

    train_dl = DataLoader(train, batch_size=512, shuffle=True)
    
    resnet_model = ResNetModel().to(device)
    cnn_model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    best_model1 = train_function(model = resnet_model, 
                                        train_dataloader=train_dl,
                                        loss_fn=loss_fn,
                                        device=device)
    
    best_model2 = train_function(model = cnn_model, 
                                        train_dataloader=train_dl,
                                        loss_fn=loss_fn,
                                        device=device)
    
    torch.save(best_model1, 'quick_lol1.pth')
    save_data('quick_lol1.pth')

    torch.save(best_model2, 'quick_lol2.pth')
    save_data('quick_lol2.pth')