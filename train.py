import torch
import visdom
from torch import optim, nn
from Data_Pre import Data
from utils import Flatten
from torch.utils.data import DataLoader
from torchvision.models import resnet18

batchsz = 32
lr = 1e-4
epochs =10

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Data('train_data', 224, mode='train')
val_db = Data('train_data', 224, mode='val')
test_db = Data('train_data', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=4)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=4)

viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total


def main():
    trained_model=resnet18(pretrained=True)  
    model = nn.Sequential(*list(trained_model.children())[:-1],  
                          Flatten(),
                          nn.Linear(512,6)
                          ).to(device)
    # x=torch.randn(2,3,224,224)
    # print(model(x).shape)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0, 0
    global_step = 0
    # viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    # viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

    viz.line([0.], [0.], win='loss', opts=dict(title='Loss on Training Data',xlabel='Epochs',ylabel='Loss'))
    viz.line([0.], [0.], win='val_acc', opts=dict(title='Accuracy on Training Data',xlabel='Epochs',ylabel='Accuracy'))

    ## loss和val_acc放在一个图中
    viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='Loss on Training Data and Accuracy on Training Data',
                                                       xlabel='Epochs', ylabel='Loss and Accuracy',
                                                       legend=['loss', 'val_acc']))

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append') # 更新loss

            viz.line([[loss.item(), evalute(model, val_loader)]],[global_step], win='test', update='append') # 更新loss和val_acc

            global_step += 1

        if epoch % 1 == 0:
            print('第 '+str(epoch+1)+' 批'+' training……')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best_trans.mdl')

                viz.line([val_acc], [global_step], win='val_acc', update='append') # 更新accuracy

    print('最好的准确率:', best_acc, '最好的批次:',(best_epoch+1))

    # model.load_state_dict(torch.load('best.mdl'))
    torch.save(model, 'model.pkl')
    print('正在加载模型……')

    test_acc = evalute(model, test_loader)
    print('测试准确率:', test_acc)
    print('保存最好效果模型成功！')


if __name__ == '__main__':
    main()