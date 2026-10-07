from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None, mode='train'):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.mode = mode  # Added mode to distinguish train/val

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError(f"image: {img_path} isn't RGB mode.")
        label = self.images_class[item]

        # *** CORRECTED LOGIC ***
        if self.mode == 'train':
            # For contrastive learning, return two views
            view1 = self.transform(img)
            view2 = self.transform(img)
            return view1, view2, label
        else: # For 'val' or 'test' mode
            # For standard evaluation, return one view
            return self.transform(img), label

    @staticmethod
    def collate_fn(batch):
        # This function now correctly handles both training and validation cases
        # based on the number of items returned by __getitem__

        # Check if the first item in the batch is for training (3 elements) or validation (2 elements)
        if len(batch[0]) == 3:  # Training mode: (view1, view2, label)
            views1, views2, labels = zip(*batch)
            views1 = torch.stack(views1, dim=0)
            views2 = torch.stack(views2, dim=0)
            labels = torch.as_tensor(labels)
            return (views1, views2), labels
        else:  # Validation mode: (image, label)
            images, labels = zip(*batch)
            images = torch.stack(images, dim=0)
            labels = torch.as_tensor(labels)
            return images, labels