# Dataset preparation script
import os
import shutil
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/dataset"

def prepare(split=(0.8, 0.1, 0.1)):
    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    os.makedirs(OUT_DIR, exist_ok=True)

    for part in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, part), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(RAW_DIR, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png'))]
        
        if len(imgs) == 0:
            continue

        train, temp = train_test_split(imgs, train_size=split[0], random_state=42)
        val, test = train_test_split(temp, test_size=split[2] / (split[1] + split[2]), random_state=42)

        for p, arr in zip(["train", "val", "test"], [train, val, test]):
            out_cls_dir = os.path.join(OUT_DIR, p, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            
            for src in arr:
                dst = os.path.join(out_cls_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    print("✅ Dataset prepared successfully at:", OUT_DIR)


if __name__ == '__main__':
    prepare()
