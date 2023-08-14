import os
import torch
from PIL import Image


class ImageDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root_path, img_id_list, transform=None):
        super().__init__()
        self.root_path = root_path
        self.img_id_list = img_id_list

        self.t = transform

    def __getitem__(self, i):
        img_id = self.img_id_list[i]
        img_path = os.path.join(
            self.root_path,
            img_id[0], img_id[1], img_id[2],
            img_id+".jpg"
        )
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.t is not None:
            img = self.t(img)
        
        return img, img_id

    def __len__(self):
        return len(self.img_id_list)