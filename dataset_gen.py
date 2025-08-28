import os
import shutil
import random
import kagglehub

# ====================================================
# 1. Download Dataset
# ====================================================
path = kagglehub.dataset_download("warcoder/potato-leaf-disease-dataset")
print("Path to raw dataset files:", path)

# Original dataset base (from Kaggle dataset)
base = os.path.join(path, "Potato Leaf Disease Dataset in Uncontrolled Environment")

# New dataset base in repo root
root = os.path.abspath(".")  # repo root
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")

# Classes to include (excluding "Nematode")
classes = ["Bacteria", "Fungi", "Healthy", "Pest", "Phytopthora", "Virus"]

# ====================================================
# Make new directories
# ====================================================
for split in [train_dir, test_dir]:
    if not os.path.exists(split):
        os.makedirs(split)
    for cls in classes:
        os.makedirs(os.path.join(split, cls), exist_ok=True)

# ====================================================
# Split data (200 for train, rest for test)
# ====================================================
for cls in classes:
    src_folder = os.path.join(base, cls)
    images = os.listdir(src_folder)
    random.shuffle(images)

    train_imgs = images[:200]
    test_imgs = images[200:]

    # Copy train
    for img in train_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(train_dir, cls, img)
        )

    # Copy test
    for img in test_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(test_dir, cls, img)
        )

print("✅ Dataset prepared at:")
print(f"Train folder: {train_dir}")
print(f"Test folder: {test_dir}")
