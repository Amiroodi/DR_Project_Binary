import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
import cv2
import numpy as np

APTOS_train_image_folder = "../APTOS/resized_train_15"
APTOS_train_csv_file = "../APTOS/labels/trainLabels15.csv"  
# APTOS_train_csv_file = "../APTOS/labels/down_train_15.csv"  


# APTOS_test_image_folder = "../APTOS/resized_test_15"
# APTOS_test_csv_file = "../APTOS/labels/testLabels15.csv"  

APTOS_test_image_folder = "../APTOS/resized_train_19"
APTOS_test_csv_file = "../APTOS/labels/trainLabels19.csv" 

NUM_WORKERS = 8

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img

def circle_crop(img, sigmaX = 30):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 
    
class LoadDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image filename and label from the DataFrame
        img_name = self.df.iloc[idx, 0]  # Assuming first column is filename
        label = self.df.iloc[idx, 1]  # Assuming second column is label (0-4)
        
        if label >=1: label = 1.0
        
        # Load image
        img_path = os.path.join(self.image_folder, img_name) + '.jpg'
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = circle_crop(image) 

        # Apply transformations
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label
    
def create_train_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    train_dataset = LoadDataset(APTOS_train_image_folder, APTOS_train_csv_file, transform=transform)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        train_dataset = Subset(train_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'DR']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return train_dataloader, class_names

def create_test_dataloader(
    transform: A.Compose,
    batch_size: int, 
    shrink_size=None,
    num_workers: int=NUM_WORKERS,
    ):
  
    test_dataset = LoadDataset(APTOS_test_image_folder, APTOS_test_csv_file, transform=transform)

    # Shrinking dataset size for test purposes
    if shrink_size is not None:
        test_dataset = Subset(test_dataset, range(shrink_size))

    # Get class names
    class_names = ['No DR', 'DR']

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    return test_dataloader, class_names