import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa 5 thư mục con (các lớp)
source_dir = './Img_detected'

# Đường dẫn đến thư mục chứa thư mục con mới (train, test, valid)
target_dir = './Data'

# Tạo thư mục train, test và valid
for split in ['train', 'test']:
    split_dir = os.path.join(target_dir, split)
    os.makedirs(split_dir, exist_ok=True)

# Lặp qua từng thư mục con trong thư mục nguồn và phân chia dữ liệu
for class_folder in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_folder)
    train_split, test_split = train_test_split(os.listdir(class_path), test_size=0.4)

    for i, split in enumerate(['train', 'test']):
        for file in locals()[f'{split}_split']:
            src_path = os.path.join(class_path, file)
            dst_path = os.path.join(target_dir, split, class_folder, file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)