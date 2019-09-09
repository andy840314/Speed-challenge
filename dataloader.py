import torch
from torch.utils.data import DataLoader
import cv2
def collate(data):
    label = torch.tensor([label for label,_ in data])
    raw_imgs_batch = [img for _,img in data]
    imgs_batch = list()
    for imgs in raw_imgs_batch:
        imgs_list = list()
        for i, path in enumerate(imgs):
            #img = img[190:350, 100:520]
            img = cv2.imread(path)
            img = img[10:, 30:500-50]
            #img = torch.from_numpy(img)
            img = torch.from_numpy(cv2.resize(img, (200, 66)))
            imgs_list.append(img)
            #imgs_list.append(torch.from_numpy(cv2.resize(cv2.imread(path), (300, 102))))
            #idx = float(path[-9:-4])
            #temp=torch.zeros(66,200,3)
            #print(temp.size())
            #imgs_list.append(temp.fill_(idx))
        #print(imgs)
        imgs_batch.append(torch.stack(imgs_list,dim=0))
    imgs_batch = torch.stack(imgs_batch, dim=0)
    return label,(imgs_batch.float()/255).permute(0, 4, 1, 2, 3)

def get_loader(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = 5,
                            collate_fn = collate)
    return dataloader
