import os
import time
import random
import numpy as np
import pandas as pd
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

CLASS_NAMES = ['EarlyPreB', 'PreB', 'ProB', 'benign']
IMG_SIZE    = (224, 224)
RANDOM_SEED = 88


# -----------------------------------------------------------------------------
# 1. Folder structure
# -----------------------------------------------------------------------------

def create_temp_folders(base_path='data/tmp'):

    folders = {
        'prepared_data': CLASS_NAMES,
        'prepared_test': CLASS_NAMES
    }
    for parent, subfolders in folders.items():
        for sub in subfolders:
            os.makedirs(os.path.join(base_path, parent, sub), exist_ok=True)
    print('All temporary folders created successfully!')


# -----------------------------------------------------------------------------
# 2. Train / test split
# -----------------------------------------------------------------------------

def prepare_data_splits(data_dir, train_ratio=0.90, random_seed=RANDOM_SEED):

    data_list = sorted(list(paths.list_images(data_dir)))
    labels    = [os.path.basename(os.path.dirname(p)) for p in data_list]

    train_list, test_list = train_test_split(
        data_list,
        train_size=train_ratio,
        stratify=labels,
        shuffle=True,
        random_state=random_seed
    )

    print(f'Number of training samples : {len(train_list)}')
    print(f'Number of testing samples  : {len(test_list)}')

    from collections import Counter
    train_labels = [os.path.basename(os.path.dirname(p)) for p in train_list]
    print(f'Train distribution: {dict(Counter(train_labels))}')

    return train_list, test_list


# -----------------------------------------------------------------------------
# 3. Test data processing (resize only)
# -----------------------------------------------------------------------------

def process_and_save_test_data(test_list, output_dir='data/tmp/prepared_test'):

    print('Processing test data...')

    for p, img_path in enumerate(test_list):
        img   = cv2.imread(img_path)
        img   = cv2.resize(img, IMG_SIZE)
        label = os.path.basename(os.path.dirname(img_path))
        save_path = os.path.join(output_dir, label, f'{label}_{p:05d}.png')
        cv2.imwrite(save_path, img)

    print(f'Test data processing completed! Processed {len(test_list)} images.')


# -----------------------------------------------------------------------------
# 4. K-Means segmentation
# -----------------------------------------------------------------------------

def kmeans_segment(img, k=3, attempts=5):

    img_flat = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        img_flat, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )
    centers   = centers.astype(np.uint8)
    segmented = centers[labels.flatten()]
    return segmented.reshape(img.shape)


# -----------------------------------------------------------------------------
# 5. Training data processing: K-Means segmentation + oversampling
# -----------------------------------------------------------------------------

def process_and_save_train_data_with_segmentation(
    train_list,
    output_dir='data/tmp/prepared_data',
    target_count=None,
    kmeans_k=3
):

    from collections import Counter

    print('=' * 60)
    print('Processing training data: K-Means segmentation + oversampling...')
    print('=' * 60)

    tic = time.perf_counter()

    # ── Distribusi kelas ──────────────────────────────────────────────────────
    labels_list  = [os.path.basename(os.path.dirname(p)) for p in train_list]
    class_counts = Counter(labels_list)
    max_count    = max(class_counts.values())

    # Kelompokkan path per kelas
    class_paths = {cls: [] for cls in class_counts}
    for img_path in train_list:
        lbl = os.path.basename(os.path.dirname(img_path))
        class_paths[lbl].append(img_path)

    # Basis count setelah duplikasi K-Means = 2 × jumlah asli per kelas
    base_counts = {cls: 2 * len(paths) for cls, paths in class_paths.items()}
    max_base    = max(base_counts.values())

    # Tentukan target: pakai argumen jika diberikan, fallback ke max_base
    # Kelas yang basisnya > effective_target TIDAK dipotong
    if target_count is None:
        effective_target = max_base
    else:
        effective_target = int(target_count)

    print(f'Class distribution (asli) : {dict(class_counts)}')
    print(f'Max count kelas (asli)    : {max_count} images')
    print(f'Basis setelah K-Means 2x  : {base_counts}')
    print(f'Target per class          : {effective_target} images'
          + (' (custom)' if target_count is not None else ' (= max basis)'))
    print(f'K-Means cluster (k)       : {kmeans_k}')
    print('-' * 60)

    # Daftar augmentasi top-up (di luar orig & kmeans, yang selalu dibuat)
    AUGMENTATIONS = {
        'flip_h': lambda img: cv2.flip(img, 1),
        'flip_v': lambda img: cv2.flip(img, 0),
        'rot90':  lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        'rot180': lambda img: cv2.rotate(img, cv2.ROTATE_180),
    }
    aug_names = list(AUGMENTATIONS.keys())

    random.seed(RANDOM_SEED)
    global_counter = 0

    for cls, img_paths in class_paths.items():
        n_orig = len(img_paths)

        # ── Step 1: simpan gambar asli + twin hasil segmentasi K-Means ────────
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, IMG_SIZE)

            # 1a. Gambar asli
            save_path = os.path.join(output_dir, cls,
                                     f'{cls}_{global_counter:05d}_orig.png')
            cv2.imwrite(save_path, img)
            global_counter += 1

            # 1b. Twin hasil segmentasi K-Means
            seg_img = kmeans_segment(img, k=kmeans_k)
            save_path = os.path.join(output_dir, cls,
                                     f'{cls}_{global_counter:05d}_kmeans.png')
            cv2.imwrite(save_path, seg_img)
            global_counter += 1

        n_base = 2 * n_orig

        # ── Step 2: top-up dengan flip/rotate jika basis < target ─────────────
        n_extra = max(effective_target, n_base) - n_base
        if n_extra > 0:
            # Pilih (img_path, aug_name) secara acak dengan replacement
            extra_paths = random.choices(img_paths, k=n_extra)
            extra_augs  = random.choices(aug_names, k=n_extra)

            for img_path, aug_name in zip(extra_paths, extra_augs):
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                aug_img   = AUGMENTATIONS[aug_name](img)
                save_path = os.path.join(output_dir, cls,
                                         f'{cls}_{global_counter:05d}_{aug_name}.png')
                cv2.imwrite(save_path, aug_img)
                global_counter += 1

        if global_counter % 500 < 50:
            print(f'   Saved {global_counter} images so far...')

    toc = time.perf_counter()

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    print('=' * 60)
    print('PREPROCESSING COMPLETED!')
    print(f'Time taken: {((toc - tic)/60):.2f} minutes')
    total = 0
    for cls in CLASS_NAMES:
        cls_path = os.path.join(output_dir, cls)
        if os.path.isdir(cls_path):
            n = len(os.listdir(cls_path))
            total += n
            orig = class_counts.get(cls, 0)
            print(f'   {cls:<12}: {n:>5} images '
                  f'(dari {orig} asli, {orig} orig + {orig} kmeans twin, '
                  f'+{n - 2 * orig} top-up flip/rotate)')
    print(f'Total: {total} images')
    print('=' * 60)


# -----------------------------------------------------------------------------
# 5. Build DataFrames
# -----------------------------------------------------------------------------

def create_dataframes(train_dir='data/tmp/prepared_data',
                      test_dir='data/tmp/prepared_test',
                      random_seed=RANDOM_SEED):

    def _collect(base_dir):
        filenames, labels = [], []
        for cls in sorted(os.listdir(base_dir)):
            cls_path = os.path.join(base_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filenames.append(os.path.join(cls_path, fname))
                    labels.append(cls)
        return pd.DataFrame({'filepath': filenames, 'label': labels})

    train_df = _collect(train_dir).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df  = _collect(test_dir).reset_index(drop=True)

    print('DataFrames created successfully!')
    print(f'Training samples : {len(train_df)}')
    print(f'Test samples     : {len(test_df)}')
    print('\nClass distribution in training data:')
    print(train_df['label'].value_counts())

    return train_df, test_df


# -----------------------------------------------------------------------------
# 6. ImageDataGenerators
# -----------------------------------------------------------------------------

def create_data_generators(train_df, valid_df, test_df,
                            batch_size=32, img_size=IMG_SIZE):

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    train_gen_config = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.1,
        brightness_range=[0.85, 1.15],
        fill_mode='nearest'
    )

    # ── Val/Test generator ───────────────────────────────────────────────────
    val_test_gen_config = ImageDataGenerator(rescale=1. / 255)

    common_kwargs = dict(
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        color_mode='rgb'
    )

    train_gen = train_gen_config.flow_from_dataframe(
        train_df, shuffle=True, batch_size=batch_size, seed=RANDOM_SEED,
        **common_kwargs
    )

    valid_gen = val_test_gen_config.flow_from_dataframe(
        valid_df, shuffle=True, batch_size=batch_size, seed=RANDOM_SEED,
        **common_kwargs
    )

    # Test: load semua sekaligus
    test_gen = val_test_gen_config.flow_from_dataframe(
        test_df, shuffle=False, batch_size=len(test_df),
        **common_kwargs
    )

    print(f'\nData generators ready:')
    print(f'   Train  batches : {len(train_gen)}')
    print(f'   Valid  batches : {len(valid_gen)}')
    print(f'   Test   batch   : 1 (all {len(test_df)} samples at once)')

    return train_gen, valid_gen, test_gen