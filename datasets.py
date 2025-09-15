'''
@saulzhang
The implementation code of dataset loader class in the the paper 
"Predicting Future Frames using Retrospective Cycle GAN" with Pytorch
data: Nov,17,2019
'''

import glob
import random
import os
import torchvision.transforms as transforms
import torch
import random
import pickle

from torch.utils.data import Dataset
from PIL import Image

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def video_load(video_path,nt=5):
     video_info = []
     video_folder = os.listdir(video_path)
     for video_name in video_folder:
         num_img = len(os.listdir(os.path.join(video_path,video_name)))
         for j in range(num_img-nt+1):
             index_set = []
             for k in range(j, j + nt):
                 index_set.append(os.path.join(video_path,os.path.join(video_name,str(k)+".jpg")))
             video_info.append(index_set)
     return video_info

class ImageTrainDataset(Dataset):
    def __init__(self, video_pkl_file, transforms_=None,nt=10):
        self.video_pkl_file = video_pkl_file
        self.transform = transforms.Compose(transforms_)
        self.nt = nt
        train_pkl_file = open(video_pkl_file, 'rb')
        self.data = pickle.load(train_pkl_file)
        # self.data = video_load(video_path,nt)
        self.data.sort()
        # if not os.path.exists("./dataset/pickle_data/train_data.pkl"):#引入该模块的意义在于使得test和train的划分可以持久化的保存下来，方便反复进行实验
        #     print("create folder './dataset/pickle_data'")
        #     os.makedirs("./dataset/pickle_data")
        #     train_dataset_output = open('./dataset/pickle_data/train_data.pkl', 'wb')
        #     pickle.dump(self.data, train_dataset_output)
        #     train_dataset_output.close()
        # else:
        #     train_pkl_file = open('./dataset/pickle_data/train_data.pkl', 'rb')
        #     self.data = pickle.load(train_pkl_file)

    def __getitem__(self, index):
        
        root =  '../autodl-fs/cal/train_img/'
        start = 6941- self.nt
        end = 6941
        frame_seq = []
#         print('index 0',index)
        index_0 = index -self.nt
        
        #if random.random() < 0.3:#以0.3的概率对整个图片序列进行水平翻转
           # Flip_p = 1.0#翻转
#         print('index 1:',index_0)
        for img_name in self.data[index_0+self.nt:index_0+self.nt*2]:
#             print('name',img_name)
            img = Image.open(root+img_name)
            img = self.transform(img)
            frame_seq.append(img)
        frame_seq = torch.stack(frame_seq, 0)

        if index_0 == end-self.nt*2:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+1:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+2:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+3:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+4:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+5:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+6:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+7:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+8:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+9:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+10:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        return frame_seq

    def __len__(self):

        return len(self.data)

class ImageTestDataset(Dataset):
    def __init__(self, video_pkl_file, transforms_=None,nt=10):
        self.video_pkl_file = video_pkl_file
        self.transform = transforms.Compose(transforms_)
        self.nt = nt
        # self.data = video_load(video_path,nt)
        train_pkl_file = open(video_pkl_file, 'rb')
        self.data = pickle.load(train_pkl_file)
        self.data.sort()
        # if not os.path.exists("./dataset/pickle_data/test_data.pkl"):#引入该模块的意义在于使得test和train的划分可以持久化的保存下来，方便反复进行实验
        #     if not os.path.exists("./dataset/pickle_data/"):
        #         print("create folder './dataset/pickle_data'")
        #         os.makedirs("./dataset/pickle_data")
        #     test_dataset_output = open('./dataset/pickle_data/test_data.pkl', 'wb')
        #     pickle.dump(self.data, test_dataset_output)
        #     test_dataset_output.close()
        # else:
        #     test_pkl_file = open('./dataset/pickle_data/test_data.pkl', 'rb')
        #     self.data = pickle.load(test_pkl_file)

    def __getitem__(self, index):
        
        index_0 = index - self.nt
        root = '../autodl-fs/cal/test_img/'
        start = 2133 - self.nt
        end = 2133
        frame_seq = []
#         print('index 0',index)
        index_0 = index -self.nt
        
        #if random.random() < 0.3:#以0.3的概率对整个图片序列进行水平翻转
           # Flip_p = 1.0#翻转
        #print('index 1:',index_0)
        for img_name in self.data[index_0+self.nt:index_0+self.nt*2]:
#             print('name',img_name)
            img = Image.open(root+img_name)
            img = self.transform(img)
            frame_seq.append(img)
        frame_seq = torch.stack(frame_seq, 0)

        if index_0 == end-self.nt*2:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+1:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+2:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+3:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+4:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+5:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+6:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+7:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+8:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+9:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        return frame_seq

    def __len__(self):
        return len(self.data)

class ImageValDataset(Dataset):
    def __init__(self, video_pkl_file, transforms_=None,nt=10):
        self.video_pkl_file = video_pkl_file
        self.transform = transforms.Compose(transforms_)
        self.nt = nt
        # self.data = video_load(video_path,nt)
        train_pkl_file = open(video_pkl_file, 'rb')
        self.data = pickle.load(train_pkl_file)
        # if not os.path.exists("./dataset/pickle_data/test_data.pkl"):#引入该模块的意义在于使得test和train的划分可以持久化的保存下来，方便反复进行实验
        #     if not os.path.exists("./dataset/pickle_data/"):
        #         print("create folder './dataset/pickle_data'")
        #         os.makedirs("./dataset/pickle_data")
        #     test_dataset_output = open('./dataset/pickle_data/test_data.pkl', 'wb')
        #     pickle.dump(self.data, test_dataset_output)
        #     test_dataset_output.close()
        # else:
        #     test_pkl_file = open('./dataset/pickle_data/test_data.pkl', 'rb')
        #     self.data = pickle.load(test_pkl_file)

    def __getitem__(self, index):
        root = '../autodl-fs/cal/val_img/'
        start = 568 - self.nt
        end = 568
        frame_seq = []
#         print('index 0',index)
        index_0 = index -self.nt
        
        #if random.random() < 0.3:#以0.3的概率对整个图片序列进行水平翻转
           # Flip_p = 1.0#翻转
        #print('index 1:',index_0)
        for img_name in self.data[index_0+self.nt:index_0+self.nt*2]:
#             print('name',img_name)
            img = Image.open(root+img_name)
            img = self.transform(img)
            frame_seq.append(img)
        frame_seq = torch.stack(frame_seq, 0)

        if index_0 == end-self.nt*2:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+1:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+2:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+3:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+4:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0) 
        elif index_0 == end-self.nt*2+5:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+6:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+7:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+8:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        elif index_0 == end-self.nt*2+9:
              frame_seq = []
              for img_name in self.data[start:end]:

                    img = Image.open(root+img_name)
                    img = self.transform(img)
                    frame_seq.append(img)
              frame_seq = torch.stack(frame_seq, 0)
        return frame_seq

    def __len__(self):
        return len(self.data)
