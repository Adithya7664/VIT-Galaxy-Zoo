import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GalaxyZooDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform=None, labeled=False):
        self.img_dir = img_dir
        self.transform = transform
        self.labeled = labeled

        # collect all jpg paths sorted by filename
        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.jpg')
        ])

        if labeled:
            df = pd.read_csv(csv_path)
            prob_cols = [c for c in df.columns if c != 'GalaxyID']
            self.ids = df['GalaxyID'].tolist()
            self.labels = torch.tensor(df[prob_cols].values.astype('float32'))
            # build id->label index for fast lookup
            self.id_to_label = {gid: i for i, gid in enumerate(self.ids)}
            # keep only images that exist in the CSV
            self.img_paths = [
                p for p in self.img_paths
                if int(os.path.splitext(os.path.basename(p))[0]) in self.id_to_label
            ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labeled:
            galaxy_id = int(os.path.splitext(os.path.basename(img_path))[0])
            label = self.labels[self.id_to_label[galaxy_id]]
            return image, label

        return image
