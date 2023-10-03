from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

def loaddata(task, if_tb):
    writer = None
    if task == 'M':
        if if_tb:
            writer = SummaryWriter(comment = '-Mni')
        hyperparams = [100, 784, 10, 1e-3, 20, 'MNIST', 1e-3]
        train_dataset = dsets.MNIST(root = './data/mnist', train = True, transform = transforms.ToTensor(), download = True)
        test_dataset = dsets.MNIST(root = './data/mnist', train = False, transform = transforms.ToTensor())
        train_loader = DataLoader(dataset = train_dataset, batch_size = hyperparams[0], shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = hyperparams[0], shuffle = False)
    return writer, hyperparams, train_dataset, test_dataset, train_loader, test_loader
