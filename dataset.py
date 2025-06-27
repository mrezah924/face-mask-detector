import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = ['with_mask', 'without_mask']
        self.transform = transform
        self.images = []
        for label in self.classes:
            folder = os.path.join(root_dir, label)
            for img in os.listdir(folder):
                self.images.append((os.path.join(folder, img), self.classes.index(label)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label