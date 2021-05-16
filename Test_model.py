import sys
import torch
from PIL import Image
from torchvision import transforms

classes = ('harmful','kitch','others','recyc')
device = torch.device('cuda')
transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])
def prediect(img_path):
    net=torch.load('model.pkl')
    net=net.to(device)
    torch.no_grad()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print(classes[predicted[0]])

if __name__ == '__main__':
    #prediect('./test/4.jpg')
    prediect(sys.argv[1])
