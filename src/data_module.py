import os
import cv2
from torch.utils.data import Dataset
from utils import load_config




# ===============================================================
# DATA PREPROCESSING AND AUGMENTATION FUNCTIONS
# ===============================================================




# ===============================================================
# LOADING DATASET
# ===============================================================
class GeoDataset(Dataset):
    def __init__(self, data_config: dict):
        self.data_config = data_config
        self.raw_data_dir = data_config['RAW_DATA_DIR']
        self.labels = [] 
        
        raw_images_data, raw_images_path = self._load_raw_images_and_labels()
        
        dataset = {
            'image': [],
            'path': [],
            'class': [],
            'label_idx': []
        }

        for img_data, path in zip(raw_images_data, raw_images_path):
            class_name = os.path.basename(os.path.dirname(path))
            
            if class_name not in self.labels:
                self.labels.append(class_name)
            
            dataset['image'].append(img_data)
            dataset['path'].append(path)
            dataset['class'].append(class_name)

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.labels))}
        self.dataset = dataset
        

    def _load_raw_images_and_labels(self):
        all_images_data = []
        all_images_path = []
        for class_name in os.listdir(self.raw_data_dir):
            class_dir = os.path.join(self.raw_data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, file_name)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        all_images_data.append(img)
                        all_images_path.append(img_path)

        return all_images_data, all_images_path
    
    def __len__(self):
        return len(self.dataset['image'])

    def __getitem__(self, idx):
        image = self.dataset['image'][idx]
        class_name = self.dataset['class'][idx]
        label = self.class_to_idx[class_name]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.labels)




'''
data_config = load_config("config/data_config.yml")
dataset = GeoDataset(data_config=data_config)
print(f"Loaded dataset with {len(dataset.dataset['image'])} images.")
'''