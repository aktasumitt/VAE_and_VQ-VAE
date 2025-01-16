from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
from glob import glob
from PIL import Image


# Loading Dataset
def Loading_Dataset(dataset_dir):
    
    img_path_list=[]
    
    for i,folder in enumerate(glob(pathname=dataset_dir+"\*")):
        for _,img in enumerate(glob(folder+"\*")):
            img_path_list.append(img)
        
        if i==3: break
            
    print(F"\n...{len(img_path_list)} Data were Loaded.\n")    
    return img_path_list
        


# Create Dataset 
class Datasets(Dataset):
    def __init__(self,full_img_list):
        super(Datasets,self).__init__()
        
        self.transformer=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,),(0.5,))])
        self.full_img_list=full_img_list
        
    def __len__(self):
        return len(self.full_img_list)

    def __getitem__(self, index):
        
        img_path=self.full_img_list[index]
        image=Image.open(img_path)
        image=self.transformer(image)
                
        return image
    
# Random Split    
def Random_split(dataset,test_split):
    
    test_size=int(len(dataset)*test_split)
    train_Size=int(len(dataset)-(test_size))
    
    train,test=random_split(dataset,[train_Size,test_size])
    
    return train,test


# Create Dataloader
def Dataloader(train=None,test=None,batch_size=None):
    
    train_load=DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=True)
    test_load=DataLoader(dataset=test,batch_size=batch_size,shuffle=False,drop_last=True)
    
    print("Dataloaders were created...\n  ")
    
    return train_load,test_load
