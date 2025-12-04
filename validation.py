import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
train_dir = r"C:\Users\ksrak\OneDrive\Desktop\FINAL_YEAR_PROJECT\EXPLAINABLE_AI_BRAIN_TUMOR\DATASET\archive\Training"
val_dir = r"C:\Users\ksrak\OneDrive\Desktop\FINAL_YEAR_PROJECT\EXPLAINABLE_AI_BRAIN_TUMOR\DATASET\archive\Validation"

# Create validation folder if not exists
os.makedirs(val_dir, exist_ok=True)

# Loop through each class folder in Training
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    
    if os.path.isdir(class_path):
        # Create same class folder in Validation
        val_class_path = os.path.join(val_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        # Get all images in this class
        images = os.listdir(class_path)
        
        # Split into train and validation (80/20)
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        # Move validation images
        for img in val_imgs:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(val_class_path, img)
            shutil.copy(src_path, dst_path)   # use copy instead of move, so Training is intact

print(" Validation dataset created and saved successfully at:", val_dir)
