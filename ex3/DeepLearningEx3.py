import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm

###
cuda = True
seed = 42
use_cuda = cuda and torch.cuda.is_available()

# Set seed
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# Handel GPU stochasticity
torch.backends.cudnn.enabled = use_cuda
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_cuda else "cpu")
###
original_size = 96
cropped_size = 64
num_channels = 3
num_classes = 10

##

mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize(mean, std)])

train_transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(cropped_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                     transforms.Normalize(mean, std)
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

trainset = torchvision.datasets.STL10(root='./ex3/data', split='train', download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True, num_workers=1)


test_transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop(cropped_size),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
testset = torchvision.datasets.STL10(root='./ex3/data', split='test',
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=1)

###

classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

###

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

imgs_dict = dict(zip(classes, [[],[],[],[],[],[],[],[],[],[]]))
counters_dict = dict(zip(classes, [0]*10))

while not all(v >= 4 for v in counters_dict.values()):
    images, labels = dataiter.next()
    for image, label in zip(images, labels):
        img_class = classes[label]
        if(counters_dict[img_class] < 4):
            imgs_dict[img_class].append(image)
            counters_dict[img_class] += 1


###
    
# show images

# for l,imgs in imgs_dict.items():
#     print(l)
#     imshow(torchvision.utils.make_grid(imgs, nrow=4))
###

def train(net, trainloader, num_epochs=50, validation_ratio = 0.2):
	criterion = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	optimizer = optim.Adam(net.parameters(), lr=0.0001)
	epoch_pbar =  tqdm(range(num_epochs))
	assert 0.05 < validation_ratio < 0.4
	save_for_val_every =  int(1/validation_ratio)
	val_inputs, val_labels = [],[]
	
	for epoch in epoch_pbar:  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
            
			if i % save_for_val_every == save_for_val_every - 1:
				# save for validation
				val_inputs.append(inputs)
				val_labels.append(labels)
			else:
				#train

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0

		#validation
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in zip(val_inputs, val_labels):
				outputs = net(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()


		epoch_pbar.set_postfix({'epoch': epoch, 'train loss': running_loss, 'val_accuracy': (100 * correct / total)})
		# epoch_pbar.set_postfix({'epoch': epoch, 'train accuracy': train_acc, 'train loss': train_loss, \
        #                      'val accuracy': val_acc, 'val loss': val_loss})

	print('Finished Training')

###

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' \
         % (100 * correct / total))

###

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(cropped_size*cropped_size*num_channels, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


###
net = LogisticRegression().to(device)
# summary(net, (num_channels, cropped_size, cropped_size))
###
train(net, trainloader)  
###
test(net, testloader)
###
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
	        nn.Linear(cropped_size*cropped_size*num_channels, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
			nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),
			nn.Linear(512, 512),
            nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),
			nn.Linear(512, 512),
            nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
###
net = FullyConnectedNN().to(device)
train(net, trainloader)  
###
test(net, testloader)
###
class ConvNN(nn.Module):
	def __init__(self):
		super(ConvNN, self).__init__()
		self.feature_extraction = nn.Sequential(
			nn.Conv2d(num_channels, 128, 3),   # 64  64  3  => 31  31 
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(2),
			nn.Conv2d(128, 64, 3),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2),
		)
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(12544, 1024),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024, 512),
            nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		x = self.feature_extraction(x)
		x = self.flatten(x) 
		logits = self.linear_relu_stack(x)
		return logits
###
net = ConvNN().to(device)
# summary(net, (num_channels, cropped_size, cropped_size))
###
train(net, trainloader)  
###
test(net, testloader)
###
class MobileNetV2FetureExtNN(nn.Module):
	def __init__(self, pretrained = True):
		super(MobileNetV2FetureExtNN, self).__init__()
		self.feature_extraction = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
		self.feature_extraction.trainable = not pretrained
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(1000, 512),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(512, 256),
            nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(256, 10)
		)

	def forward(self, x):

		x = self.feature_extraction(x)
		x = self.flatten(x) 
		logits = self.linear_relu_stack(x)
		return logits	
###
net = MobileNetV2FetureExtNN().to(device)
# summary(net, (num_channels, cropped_size, cropped_size))
###
train(net, trainloader)  
###
test(net, testloader)
###
net = MobileNetV2FetureExtNN(pretrained = False).to(device)
# summary(net, (num_channels, cropped_size, cropped_size))
###
train(net, trainloader)  
###
test(net, testloader)
###
