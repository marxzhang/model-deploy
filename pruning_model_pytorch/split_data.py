import os
import random
import shutil

random.seed(0)

src_root = os.path.expanduser("~/code/data/flower_photos")
dst_root = os.path.expanduser("~/code/data/flower_data")
val_rate = 0.2

supported = [".jpg", ".JPG", ".png", ".PNG"]

classes = [cla for cla in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, cla))]
classes.sort()

for split in ("train", "val"):
    for cla in classes:
        os.makedirs(os.path.join(dst_root, split, cla), exist_ok=True)

for cla in classes:
    cla_path = os.path.join(src_root, cla)
    images = [i for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
    random.shuffle(images)
    val_count = int(len(images) * val_rate)
    val_images = set(images[:val_count])

    for img in images:
        src = os.path.join(cla_path, img)
        dst = os.path.join(dst_root, "val" if img in val_images else "train", cla, img)
        shutil.copy2(src, dst)

    print("class {}: {} total -> train {}, val {}".format(cla, len(images), len(images) - val_count, val_count))

print("done. split saved to {}".format(dst_root))