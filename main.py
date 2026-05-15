import torch

import matplotlib.pyplot as plt


from torch import nn
from torchvision import datasets

from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy


from torchinfo import summary
from PIL import Image


train_transform_fashion=transforms.Compose([
    transforms.Resize(size=(32,32)),
    transforms.TrivialAugmentWide(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,],std=[0.5,])


])

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



train_dataset=datasets.FashionMNIST(root="data",train=True,download=True,transform=train_transform_fashion,target_transform=None)
test_dataset=datasets.FashionMNIST(root="data",train=False,download=True,transform=test_transform )

image,label=train_dataset[0]
print(image.shape)
print(label)
class_names=train_dataset.classes
print(train_dataset)

print(test_dataset)
print(class_names)

plt.title(class_names[label])
image=image.permute(1,2,0)
plt.figure(figsize=(1.5,1.5))
plt.imshow(image)
plt.show()


BATCH_SIZE=32

train_dataLoader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_dataLoader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

print(len(train_dataLoader))
print(len(test_dataLoader))

print(train_dataLoader.dataset[0][0])
print(train_dataLoader.dataset[0][1])


class FashionClassifierModel(nn.Module):
    def __init__(self,input_shape:int,hidden_units:int,num_classes:int):
        super().__init__()

        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*8*8,out_features=num_classes)
        )
    def forward(self,x):
        return self.block3(self.block2(self.block1(x)))

torch.manual_seed(42)
fashion_model=FashionClassifierModel(input_shape=1,hidden_units=32,num_classes=len(class_names))
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=fashion_model.parameters(),lr=0.001)

summary(fashion_model,input_size=[32,1,32,32])


torch.manual_seed(42)
accuracy=MulticlassAccuracy(num_classes=len(class_names))
epochs=5

train_loss_values=[]
train_acc_values=[]
test_loss_values=[]
test_acc_values=[]


for epoch in range(epochs):
    total_loss=0
    total_acc=0
    for batch,(X,y) in enumerate(train_dataLoader):
        fashion_model.train()
        y_pred=fashion_model(X)
        loss=loss_fn(y_pred,y)
        total_loss+=loss.item()
        train_accuracy=accuracy(y_pred.argmax(dim=1),y)
        total_acc+=train_accuracy.item()*100
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%400==0:
            print(f"Looked at {batch}")

    total_loss/=len(train_dataLoader)
    total_acc/=len(train_dataLoader)


    train_loss_values.append(total_loss)
    train_acc_values.append(total_acc)


    test_loss=0
    test_acc=0
    fashion_model.eval()

    with torch.inference_mode():
        for (X,y) in test_dataLoader:
           test_pred=fashion_model(X)
           testLoss=loss_fn(test_pred,y)
           test_loss+=testLoss.item()
           test_accuracy=accuracy(test_pred.argmax(dim=1),y)
           test_acc+=test_accuracy.item()*100

        test_loss/=len(test_dataLoader)
        test_acc/=len(test_dataLoader)

    test_loss_values.append(test_loss)
    test_acc_values.append(test_acc)

    print(
        f"Train loss:{total_loss} ,Train accuracy:{total_acc},Test loss:{test_loss}, Test accuracy:{test_acc}")


epoch_range=range(epochs)

plt.figure(figsize=(16,5))


plt.subplot(1,2,1)
plt.plot(epoch_range,train_loss_values,label="Train Loss",color="green")
plt.plot(epoch_range,test_loss_values,label="Test Loss",color="red")
plt.title("Train Loss vs Test Loss")
plt.xlabel("Epoch")
plt.ylabel(" Loss")
plt.legend()


plt.subplot(1,2,2)
plt.plot(epoch_range,train_acc_values,label="Train Accuracy",color="green")
plt.plot(epoch_range,test_acc_values,label="Test Accuracy",color="red")
plt.title("Train Accuracy vs Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel(" Accuracy")
plt.legend()

plt.show()

image_transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

img=Image.open("ankle_bot.jpg")
img=image_transform(img)

img=img.unsqueeze(0)

fashion_model.eval()
with torch.inference_mode():
    pred=fashion_model(img)
    probs=torch.softmax(pred,dim=1)
    pred_ids=probs.argmax(dim=1).item()


    print("Predicted class:",class_names[pred_ids])












