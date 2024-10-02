from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from PIL import Image

from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True
import MCTINet_model

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}


class Net(nn.Module):
    def __init__(self , net1, net2):
        super(Net, self).__init__()
        self.net1 = net1
        self.net2 = net2


    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)

        return x




def computeSpearman(dataloader_valid, model):
    
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        for batch_idx, data in enumerate(dataloader_valid):
            inputs = data['image']
            batch_size = inputs.size()[0]
            labels = data['rating'].view(batch_size, -1)
            # labels = labels / 100.0
            if use_gpu:
                try:
                    inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                except:
                    print(inputs, labels)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs_a = model(inputs)
            ratings.append(labels.float())
            predictions.append(outputs_a.float())

    ratings_i = torch.vstack(ratings)
    predictions_i = torch.vstack(predictions)
    a = ratings_i[:,0].cpu()
    b = predictions_i[:,0].cpu()
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]
    return sp, pl

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def finetune_model():
    epochs = 50
    srocc_l = []
    plcc_l = []
    data_dir = os.path.join('D:/dataset/IQA/ChallengeDB_release/')
    images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')

    images_fold = "D:/dataset/IQA/ChallengeDB_release/"
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(10):

        # random_seed = random.randint(0, 2**32 - 1)
        # print(f"Generated random seed: {random_seed}")
        set_seed(1158064236)

        images_train, images_test = train_test_split(images, train_size = 0.8)

        train_path = images_fold + "train_image" + ".csv"
        test_path = images_fold + "test_image" + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)

        net1 = MCTINet_model.MCTINet_front()
        net2 = MCTINet_model.MCTINet_back()
        model = Net(net1=net1, net2=net2)
        model = torch.load('Meta_train.pt')
        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4,  weight_decay=0)
        model.cuda()

        spearman = 0
        plcc = 0
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)

            if epoch == 0:
                dataloader_valid = load_data('train')
                model.eval()

                sp,pl = computeSpearman(dataloader_valid, model)

                if sp > spearman:
                    spearman = sp
                print('no train srocc {:4f}'.format(sp))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train = load_data('train')
            model.train()  # Set model to training mode
            for batch_idx, data in enumerate(dataloader_train):
                inputs = data['image']
                batch_size = inputs.size()[0]
                labels = data['rating'].view(batch_size, -1)
                # labels = labels / 100.0
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    except:
                        print(inputs, labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid = load_data('test')
            model.eval()

            sp,pl= computeSpearman(dataloader_valid, model)

            if sp + pl > spearman + plcc:
                spearman = sp
                plcc = pl
                torch.save(model.cuda(),'live_finetune_test.pt')
            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}, plcc : {:4f}'.format(epoch, pl, sp, spearman,plcc))
    
        srocc_l.append(spearman)
        plcc_l.append(plcc)
    # writer.close()
    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    print('best plcc srcc : pl : {:4f}, sr : {:4f}'.format(np.max(plcc_l), np.max(srocc_l)))
    print('average plcc {:4f}'.format(np.mean(plcc_l)))
    print('average srocc {:4f}'.format(np.mean(srocc_l)))


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train'):

    data_dir = os.path.join('D:/dataset/IQA/ChallengeDB_release/')
    train_path = os.path.join(data_dir,  'train_image.csv')
    test_path = os.path.join(data_dir,  'test_image.csv')

    output_size = (224, 224)
    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='D:/dataset/IQA/ChallengeDB_release/Images',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='D:/dataset/IQA/ChallengeDB_release/Images',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))

    if mod == 'train':
        dataloader = DataLoader(transformed_dataset_train, batch_size=50,
                                  shuffle=False, num_workers=0, collate_fn=my_collate)
    else:
        dataloader = DataLoader(transformed_dataset_valid, batch_size= 50, 
                                    shuffle=False, num_workers=0, collate_fn=my_collate)

    return dataloader

finetune_model()
