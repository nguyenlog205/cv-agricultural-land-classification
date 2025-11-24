import os
import cv2
from utils import load_config
from torch.utils.data import Dataset, DataLoader, random_split
import torch



# ===============================================================
# AUGMENTATION PIPELINE
# ===============================================================
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 64

train_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    
    # --- Geometric Transformations ---
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # --- Color and Brightness Adjustments ---
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.HueSaturationValue(p=1),
    ], p=0.3),
    
    # --- Noise Addition ---
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),

    # --- Cutting Out ---
    A.CoarseDropout(
        num_holes_range=(1, 2),
        hole_height_range=(1, 20),
        hole_width_range=(1, 20),     # Thay cho max_width
        fill_value=0,
        p=0.2
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- Val/Test ---
val_transform = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class DatasetWrapper(Dataset):
    """
    Class này nhận vào một tập con (Subset) chứa ảnh gốc,
    và áp dụng Augmentation lên nó.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # --- Take original image and label from subset ---
        image, label = self.subset[idx]
        
        # --- Apply transformation if any ---
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label


# ===============================================================
# LOADING RAW DATASET
# ===============================================================
class GeoDataset(Dataset):
    def __init__(
        self, 
        data_config: dict,
    ):
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
        dataset['label_idx'] = [self.class_to_idx[cls] for cls in dataset['class']]
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




# ===============================================================
# DATALOADER
# ===============================================================
def create_dataloaders(
    data_config, 
    train_transform = train_transform, 
    val_transform = val_transform, 
    split_ratio=(0.9, 0.05, 0.05), 
    batch_size=32
):
    # --- Load Full Dataset ---
    full_dataset = GeoDataset(data_config)
    
    # --- Calculate Split Sizes ---
    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - val_size
    
    print(f"📊 Chia tập dữ liệu: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # --- Split Dataset ---
    train_subset, val_subset, test_subset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # --- Apply Transformations ---
    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset   = DatasetWrapper(val_subset,   transform=val_transform)
    test_dataset  = DatasetWrapper(test_subset,  transform=val_transform)
    
    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader