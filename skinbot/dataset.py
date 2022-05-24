import os
from torch.utils.data import Dataset
from torchvision.io import read_image

class WoundImages(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.image_fnames= os.listdir(os.path.join(root_dir, "images"))
        self.images_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.image_fnames)
    
    def __getitem(self, index):
        image_path = os.path.join(self.image_dir, self.image_fnames[index])
        image = read_image(image_path)
        label = self.image_fnames[index].split("_")[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


