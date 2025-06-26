import os
import pandas as pd
import pickle
import numpy as np
import torch
import json
from torch.utils.data import Dataset, WeightedRandomSampler, Subset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def get_dynamic_transforms(dynamic_data_augmentation):
    dynamic_data_params = dynamic_data_augmentation.split("+")

    assert dynamic_data_params[1][0] == "t"
    translation_z = int(dynamic_data_params[1][1])
    translation_x = int(dynamic_data_params[1][2])
    translation_y = int(dynamic_data_params[1][3])

    assert dynamic_data_params[2][0] == "r"
    rotation_angle = float(dynamic_data_params[2][1:])

    assert dynamic_data_params[3][0] == "n"
    noise_level = float(dynamic_data_params[3][1:])

    print(
        f"Dynamic data augmentation; translation (z,x,y): ({translation_z},{translation_x},{translation_y}); rotation: {rotation_angle}; noise: {noise_level}")

    def transform(image_2d):
        # Ensure the tensor is on the GPU
        image_2d = image_2d.to('cuda')

        # Reshape 2D image back to 3D volume (multi-channel)
        image_3d = image_2d.reshape(3, 9, 116, 9, 116)
        image_3d = image_3d.permute(0, 1, 3, 2, 4).reshape(3, 81, 116, 116)

        if translation_z > 0 or translation_x > 0 or translation_y > 0:
            # Create a mask with ones where any channel has a non-zero value
            padded_mask = F.pad((image_3d != 0).any(dim=0).float(), (5, 5, 5, 5, 5, 5), mode='constant', value=0)

            # Initialize shifts
            depth_shift = torch.randint(-translation_z, translation_z + 1, (1,), device='cuda').item()
            height_shift = torch.randint(-translation_x, translation_x + 1, (1,), device='cuda').item()
            width_shift = torch.randint(-translation_y, translation_y + 1, (1,), device='cuda').item()

            # Calculate the slicing indices
            d_start, d_end = 5 + depth_shift, 5 + depth_shift + 81
            h_start, h_end = 5 + height_shift, 5 + height_shift + 116
            w_start, w_end = 5 + width_shift, 5 + width_shift + 116

            # Check non-zero values outside the cropped mask
            outside_depth = torch.any(padded_mask[:d_start, :, :]) or torch.any(padded_mask[d_end:, :, :])
            outside_height = torch.any(padded_mask[:, :h_start, :]) or torch.any(padded_mask[:, h_end:, :])
            outside_width = torch.any(padded_mask[:, :, :w_start]) or torch.any(padded_mask[:, :, w_end:])

            # Check each axis independently using the mask
            # depth_check = torch.any(padded_mask[d_start:d_end, :, :])
            # height_check = torch.any(padded_mask[:, h_start:h_end, :])
            # width_check = torch.any(padded_mask[:, :, w_start:w_end])
            # print("padded_mask",padded_mask.shape)

            while True:
                # print(depth_shift, height_shift, width_shift)
                # Break the loop when all shifts are valid
                if not outside_depth and not outside_height and not outside_width:
                    break

                # Regenerate shifts only for offending axes
                if outside_depth:
                    depth_shift = torch.randint(-translation_z, translation_z + 1, (1,), device='cuda').item()
                    d_start, d_end = 5 + depth_shift, 5 + depth_shift + 81
                    outside_depth = torch.any(padded_mask[:d_start, :, :]) or torch.any(padded_mask[d_end:, :, :])

                if outside_height:
                    height_shift = torch.randint(-translation_x, translation_x + 1, (1,), device='cuda').item()
                    h_start, h_end = 5 + height_shift, 5 + height_shift + 116
                    outside_height = torch.any(padded_mask[:, :h_start, :]) or torch.any(padded_mask[:, h_end:, :])

                if outside_width:
                    width_shift = torch.randint(-translation_y, translation_y + 1, (1,), device='cuda').item()
                    w_start, w_end = 5 + width_shift, 5 + width_shift + 116
                    outside_width = torch.any(padded_mask[:, :, :w_start]) or torch.any(padded_mask[:, :, w_end:])

            # Apply the translation with validated shifts to all channels
            transformed_channels = []
            for channel in image_3d:
                padded_channel = F.pad(channel, (5, 5, 5, 5, 5, 5), mode='constant', value=0)
                translated_channel = padded_channel[d_start:d_end, h_start:h_end, w_start:w_end]
                # print("translated_channel",translated_channel.shape)
                transformed_channels.append(translated_channel)

            # Stack the transformed channels
            image_3d = torch.stack(transformed_channels, dim=0)

        # Apply 3D random rotation (max 5 degrees)
        if rotation_angle > 0:
            angle = torch.FloatTensor(1).uniform_(-rotation_angle, rotation_angle).to('cuda')
            for axis in [(2, 3), (1, 3), (1, 2)]:  # (H, W), (D, W), (D, H)
                image_3d = torch.rot90(image_3d, k=int(angle.item() / 90), dims=axis)

        # Add random noise
        if noise_level > 0:
            noise = torch.normal(0, noise_level, size=image_3d.shape).to('cuda')
            image_3d += noise

        # Clip to ensure the values remain valid
        # image_3d = torch.clamp(image_3d, 0, 1)

        # Reshape back to the 2D format
        transformed_image_2d = image_3d.reshape(3, 9, 9, 116, 116).permute(0, 1, 3, 2, 4).reshape(3, 1044, 1044)

        return transformed_image_2d

    return transform


def normalize_input(x):
    return (x - x.min()) / (x.max() - x.min())


def reshape_to_grid(x):
    if x.shape[0] == 110 and x.shape[1] == 110 and x.shape[2] == 68:
        # x shape: (110, 110, 68)
        # Reshape to (17, 4) grid of 110x110 images
        return x.reshape(110, 110, 17, 4).transpose(2, 0, 3, 1).reshape(1870, 440)
    elif x.shape[0] == 116 and x.shape[1] == 116 and x.shape[2] == 81:
        # x shape: (116, 116, 81)
        # Reshape to (9, 9) grid of 116x116 images
        return x.reshape(116, 116, 9, 9).transpose(2, 0, 3, 1).reshape(1044, 1044)
    else:
        raise ValueError("Input shape must be (110, 110, 68) or (116, 116, 81)")


class MRIDataset(Dataset):

    def __init__(self, data_path, input_details, device="cpu", verbose=False, target_variable=['MCI', 'AD'],
                 control_variable="CN", amyloid="C2T2"):

        self.inputs = None
        self.outputs = None
        self.subjects_id = None
        self.class_weights = None

        self.device = device
        self.data_path = data_path
        self.verbose = verbose

        self._original_target = target_variable
        self._original_control = control_variable

        self.target_variable = target_variable
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        self.control_variable = control_variable
        if isinstance(self.control_variable, str):
            self.control_variable = [self.control_variable]
        assert isinstance(self.target_variable, list)
        assert isinstance(self.control_variable, list)

        self.transform = None
        self.input_details = input_details
        self.amyloid = amyloid

        self.groups = ['CN', 'MCI', 'AD']

        if verbose:
            print(f"Loading dataset from: {data_path}")
            print(f"Input details: {input_details}")
            print(f"Device: {device}")
            print(f"Target variable: {target_variable}")
            print(f"Control variable: {control_variable}")

        self.__load_dataset()

        self.inputs = self.inputs.to(self.device)
        self.outputs = self.outputs.to(self.device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)

        if verbose:
            print(f"Dataset loaded with {len(self)} samples")
            print(f"Input shape: {self.inputs.shape}")
            print(f"Output shape: {self.outputs.shape}")

    def __len__(self):
        return len(self.outputs) if self.outputs is not None else 0

    def __getitem__(self, idx):
        if self.inputs is None or self.outputs is None:
            raise ValueError("Dataset not loaded. Call __load_dataset() first.")

        input_data = self.inputs[idx]
        output_data = self.outputs[idx]

        if self.transform:
            input_data = self.transform(input_data)

        return input_data, output_data

    def __load_dataset(self):

        # Load the dataset
        if self.amyloid == "C2T2":
            phenotypes_path = os.path.join(self.data_path, "dMRI_phenotypes.csv")
            phenotypes_df = pd.read_csv(phenotypes_path, encoding='utf-8', dtype={'sub': str})
        else:
            phenotypes_path = os.path.join(self.data_path, "amyloid_phenotypes.csv")
            phenotypes_df = pd.read_csv(phenotypes_path, encoding='utf-8', dtype={'sub': str})

        with open(os.path.join(self.data_path, self.input_details[0]), 'rb') as f:
            subjects_input = pickle.load(f)

        inputs_3d = []
        outputs = []
        subjects_id_list = []
        subjects_session = []
        subjects_amyloid = []

        for subject in subjects_input.keys():
            if subject in ["003S4288", "052S6844", "052S6844", "052S7036", "052S7037", "052S7027"]:
                continue
            for ses in subjects_input[subject].keys():
                if (subject == "037S6046" and ses == "y4") or (subject == "023S0031" and ses == "init") or (
                        subject == "941S4376" and ses == "init"):
                    continue
                if self.input_details[1] == "noddi":
                    subject_map_key = f"sub-{subject}_ses-{ses}_noddi_"
                    subject_map_1 = subjects_input[subject][ses][subject_map_key + "odi"]
                    subject_map_2 = subjects_input[subject][ses][subject_map_key + "fintra"]
                    subject_map_3 = subjects_input[subject][ses][subject_map_key + "fextra"]
                elif self.input_details[1] == "DTI":
                    subject_map_key = f"sub-{subject}_ses-{ses}_"
                    subject_map_1 = subjects_input[subject][ses][subject_map_key + "FA"]
                    subject_map_2 = subjects_input[subject][ses][subject_map_key + "AD"]
                    subject_map_3 = subjects_input[subject][ses][subject_map_key + "RD"]
                else:
                    raise ValueError("Input details must be 'noddi' or 'DTI'")

                # Find line in df matching subject and ses
                line = phenotypes_df[(phenotypes_df["sub"] == str(subject)) & (phenotypes_df["ses"] == ses)]
                if len(line) == 0:
                    if self.verbose:
                        print(f"Skipping {subject} {ses} due to missing phenotype")
                    continue
                elif len(line) > 1 and line["Group"].nunique() > 1:  # Multiple phenotypes
                    if self.verbose:
                        print(line)
                        print(f"Skipping {subject} due to multiple phenotypes")
                    continue
                group = line["Group"].values[0]

                # Find amyloid status
                if self.amyloid != "C2T2":
                    amyloid_status = line["AMYLOID_STATUS"].values[0]
                    if amyloid_status not in [0., 1.]:
                        continue
                else:
                    amyloid_status = -1

                output_label = -1  # Initialize with invalid label
                if group in self.target_variable and group in self.control_variable:
                    # Handle amyloid classification logic
                    if self.amyloid == "C0T1":
                        output_label = int(amyloid_status) # 0 if amyloid=0, 1 if amyloid=1
                    elif self.amyloid == "C1T0":
                        output_label = 1 - int(amyloid_status) # 1 if amyloid=0, 0 if amyloid=1
                    else: # C2T2 (no amyloid used for labels here) or invalid amyloid case
                        continue # Skip if invalid setup for amyloid
                elif group in self.target_variable:
                    output_label = 1
                elif group in self.control_variable:
                    output_label = 0
                else:
                    continue

                if output_label == -1:  # Skip if no valid label assigned
                    continue
                outputs.append(output_label)

                # Stack the 3D inputs
                input_3d = np.stack([subject_map_1, subject_map_2, subject_map_3], axis=0)
                if input_3d.shape[3] != 81 and subject == "341S6820":
                    input_3d = np.pad(input_3d, ((0, 0), (0, 0), (0, 0), (3, 3)), mode='constant', constant_values=0)
                if input_3d.shape[3] != 81 and subject == "023S6334":
                    input_3d = np.pad(input_3d, ((0, 0), (0, 0), (0, 0), (1, 2)), mode='constant', constant_values=0)
                inputs_3d.append(input_3d)
                subjects_id_list.append(subject)
                subjects_session.append(ses)
                subjects_amyloid.append(amyloid_status)

        # Convert to numpy arrays
        inputs_3d = np.array(inputs_3d)
        outputs = np.array(outputs)

        # Reshape and normalize the inputs
        if inputs_3d.shape[2] == 110 and inputs_3d.shape[3] == 110 and inputs_3d.shape[4] == 68:
            inputs_2d = np.zeros((len(inputs_3d), 3, 1870, 440))
        elif inputs_3d.shape[2] == 116 and inputs_3d.shape[3] == 116 and inputs_3d.shape[4] == 81:
            inputs_2d = np.zeros((len(inputs_3d), 3, 1044, 1044))
        else:
            if len(inputs_3d) == 0:
                print("Warning: No valid samples found after filtering. Dataset is empty.")
                # Initialize empty tensors to avoid errors later
                self.inputs = torch.empty((0, 3, 1044, 1044), dtype=torch.float32)  # Adjust shape if needed
                self.outputs = torch.empty((0,), dtype=torch.int64)
                self.subjects_id = np.array([])
                self.subjects_session = []
                self.subjects_amyloid = []
                self.class_weights = torch.FloatTensor([])
                return  # Exit loading process
            raise ValueError("Input shape must be (110, 110, 68) or (116, 116, 81)")

        for i in range(len(inputs_3d)):
            for j in range(3):
                inputs_2d[i, j] = normalize_input(reshape_to_grid(inputs_3d[i, j]))

        self.inputs = torch.from_numpy(inputs_2d).float()
        self.outputs = torch.from_numpy(outputs).to(torch.int64) # Keep as long for CE Loss / one-hot
        self.subjects_id = np.array(subjects_id_list)
        self.subjects_session = subjects_session
        self.subjects_amyloid = subjects_amyloid

        class_counts = np.bincount(self.outputs.numpy().astype(int).flatten())
        total_samples = np.sum(class_counts)
        self.class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
        print(f"Class weights: {self.class_weights}")

        print(f"Loaded {len(self)} subjects")
        print(f"Input shape: {self.inputs.shape}")

        # Keep outputs as class indices (LongTensor) for CrossEntropyLoss
        #self.outputs = F.one_hot(self.outputs.to(torch.int64)).float()
        print(f"Output shape: {self.outputs.shape}")


    def get_sampler(self):
        sample_weights = self.class_weights[self.outputs.cpu().numpy().astype(int).flatten()]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def compute_class_weights(self, indices=None):
        if self.outputs is None or len(self.outputs) == 0:
            print("Warning: Cannot compute class weights, dataset outputs are empty.")
            return torch.FloatTensor([])

        if indices is None:
            indices = range(len(self.outputs))
        elif len(indices) == 0:
            print("Warning: Cannot compute class weights for empty indices.")
            return torch.FloatTensor([])

        #labels = torch.argmax(self.outputs[indices], dim=1).cpu().numpy()
        labels = self.outputs[indices].cpu().numpy()

        class_counts = np.bincount(labels.astype(int).flatten())
        total_samples = np.sum(class_counts)

        # Handle the case where a class might have zero samples
        class_weights = np.zeros_like(class_counts, dtype=np.float32)
        non_zero_counts = class_counts[class_counts > 0]
        class_weights[class_counts > 0] = total_samples / (len(non_zero_counts) * non_zero_counts)

        return torch.FloatTensor(class_weights)


def generate_and_save_splits(dataset, kfold, test_ratio, seed, output_split_dir, split_filename):
    """
    Generates train/val/test splits based on subject IDs, saves them to a JSON file,
    and returns the splits. Ensures the split is generated only once per configuration.

    Args:
        dataset (MRIDataset): The full dataset object.
        kfold (int): Number of folds.
        test_ratio (float): Proportion of subjects for the fixed test set.
        seed (int): Random seed.
        output_split_dir (str): Directory to save the split file.
        split_filename (str): Filename for the JSON split file.

    Returns:
        dict: A dictionary containing the splits:
              {
                  'test_indices': [list of test indices],
                  'folds': [
                      {'train': [fold 0 train indices], 'val': [fold 0 val indices]},
                      {'train': [fold 1 train indices], 'val': [fold 1 val indices]},
                      ...
                  ],
                 'test_subjects': [list of test subject IDs],
                 'fold_subjects': [
                      {'train': [fold 0 train subject IDs], 'val': [fold 0 val subject IDs]},
                      ...
                 ]
              }
        Returns None if splitting fails.
    """
    split_file_path = os.path.join(output_split_dir, split_filename)

    # --- Check if split file already exists ---
    if os.path.exists(split_file_path):
        print(f"Loading existing splits from: {split_file_path}")
        try:
            with open(split_file_path, 'r') as f:
                splits_data = json.load(f)
            if 'test_indices' not in splits_data or 'folds' not in splits_data or len(splits_data['folds']) != kfold:
                 print(f"Warning: Loaded split file {split_file_path} seems malformed. Regenerating.")
            else:
                 # Convert loaded lists back to numpy arrays
                 splits_data['test_indices'] = np.array(splits_data['test_indices'], dtype=int)
                 for i in range(kfold):
                     splits_data['folds'][i]['train'] = np.array(splits_data['folds'][i]['train'], dtype=int)
                     splits_data['folds'][i]['val'] = np.array(splits_data['folds'][i]['val'], dtype=int)
                 return splits_data
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from existing split file: {split_file_path}. Regenerating.")
        except Exception as e:
            print(f"Error loading existing split file {split_file_path}: {e}. Regenerating.")

    print(f"Generating new splits (k={kfold}, test_ratio={test_ratio}, seed={seed})...")

    # --- Generate New Splits ---
    if not hasattr(dataset, 'subjects_id') or dataset.subjects_id is None or \
       not hasattr(dataset, 'outputs') or dataset.outputs is None:
        print("Error: Dataset is missing subject_id or outputs. Cannot generate splits.")
        return None

    subject_ids = dataset.subjects_id
    outputs = dataset.outputs.cpu().numpy()
    unique_subjects = np.unique(subject_ids)
    n_unique_subjects = len(unique_subjects)
    all_indices = np.arange(len(dataset))

    if n_unique_subjects == 0:
        print("Error: No unique subjects found in the dataset. Cannot generate splits.")
        return None
    if len(all_indices) == 0:
        print("Error: Dataset has zero samples. Cannot generate splits.")
        return None

    # --- 1. Split off Fixed Test Set based on Subjects ---
    test_indices_np = np.array([], dtype=int)
    train_val_indices_np = all_indices
    test_subjects_list = []
    train_val_subjects_list = list(unique_subjects) # Start with all

    if test_ratio > 0:
        test_size = max(1, int(test_ratio * n_unique_subjects))
        if test_size >= n_unique_subjects:
            print(f"Error: test_ratio ({test_ratio}) is too high, results in <= 0 subjects for training/validation.")
            return None

        try:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            # Split unique subjects first
            train_val_subj_idx, test_subj_idx = next(gss_test.split(X=unique_subjects, groups=unique_subjects))

            train_val_subjects = unique_subjects[train_val_subj_idx]
            test_subjects = unique_subjects[test_subj_idx]

            # Find corresponding indices in the full dataset
            train_val_indices_np = np.where(np.isin(subject_ids, train_val_subjects))[0]
            test_indices_np = np.where(np.isin(subject_ids, test_subjects))[0]

            test_subjects_list = test_subjects.tolist()
            train_val_subjects_list = train_val_subjects.tolist()

            print(f"Test split: {len(test_indices_np)} samples ({len(test_subjects)} unique subjects)")
            print(f"Train/Val pool: {len(train_val_indices_np)} samples ({len(train_val_subjects)} unique subjects)")

        except Exception as e:
            print(f"Error during GroupShuffleSplit for test set: {e}")
            return None

    elif test_ratio == 0:
        print("No fixed test set created (test_ratio=0).")

    # --- 2. Prepare for StratifiedGroupKFold on Train/Val Indices ---
    if len(train_val_indices_np) == 0:
        print("Error: No samples available for training/validation after test split.")
        return None

    y_train_val = outputs[train_val_indices_np]
    groups_train_val = subject_ids[train_val_indices_np]

    min_classes_per_group_check = pd.Series(y_train_val).groupby(groups_train_val).nunique()
    if (min_classes_per_group_check < 2).any() and kfold > 1:
        unique_labels, counts = np.unique(y_train_val, return_counts=True)
        if len(unique_labels) < 2:
            print(f"Warning: Only {len(unique_labels)} class present in the training/validation set. Stratification is not possible/meaningful.")
        elif np.min(counts) < kfold:
            print(f"Warning: The least populated class in y_train_val has only {np.min(counts)} members, which is less than n_splits={kfold}. Stratification might fail or be unreliable.")

    # --- 3. Apply StratifiedGroupKFold ---
    sgkf = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=seed)
    fold_splits = []
    fold_subject_splits = []

    try:
        splits_generator = sgkf.split(train_val_indices_np, y_train_val, groups_train_val)

        for fold_idx, (train_fold_local_idx, val_fold_local_idx) in enumerate(splits_generator):
            # Map local indices back to the original dataset's indices
            train_indices_fold_np = train_val_indices_np[train_fold_local_idx]
            val_indices_fold_np = train_val_indices_np[val_fold_local_idx]

            train_subjects_fold = np.unique(subject_ids[train_indices_fold_np]).tolist()
            val_subjects_fold = np.unique(subject_ids[val_indices_fold_np]).tolist()

            fold_splits.append({
                'train': train_indices_fold_np.tolist(),
                'val': val_indices_fold_np.tolist(),
                'val_weights':  dataset.compute_class_weights(val_indices_fold_np).tolist(),
                'train_weights': dataset.compute_class_weights(train_indices_fold_np).tolist()
            })

            fold_subject_splits.append({
                 'train': train_subjects_fold,
                 'val': val_subjects_fold
            })

            print(f"  Fold {fold_idx + 1}: Train={len(train_indices_fold_np)} samples ({len(train_subjects_fold)} unique), Val={len(val_indices_fold_np)} samples ({len(val_subjects_fold)} unique)")

    except ValueError as e:
        print(f"Error during StratifiedGroupKFold splitting: {e}")
        print("This might happen if a group has too few samples or samples of only one class, making stratification impossible with the current kfold value.")
        return None
    except Exception as e:
        print(f"Unexpected error during k-fold split generation: {e}")
        return None


    # --- 4. Prepare results and save to JSON ---
    splits_data = {
        'test_indices': test_indices_np.tolist(),
        'folds': fold_splits,
        'test_subjects': test_subjects_list,
        'fold_subjects': fold_subject_splits,
        'generation_info': {
             'kfold': kfold,
             'test_ratio': test_ratio,
             'seed': seed,
             'n_samples_total': len(dataset),
             'n_subjects_total': n_unique_subjects,
             'n_train_val_samples': len(train_val_indices_np),
             'n_train_val_subjects': len(train_val_subjects_list),
             'n_test_samples': len(test_indices_np),
             'n_test_subjects': len(test_subjects_list),
             'test_weights': dataset.compute_class_weights(test_indices_np).cpu().numpy().tolist() if len(test_indices_np) > 0 else None,
             'train_val_weights': dataset.compute_class_weights(train_val_indices_np).cpu().numpy().tolist(),
             'dataset_weights': dataset.compute_class_weights().cpu().numpy().tolist()
        }
    }

    try:
        os.makedirs(output_split_dir, exist_ok=True)
        with open(split_file_path, 'w') as f:
            json.dump(splits_data, f, indent=4)
        print(f"Splits saved successfully to: {split_file_path}")
    except Exception as e:
        print(f"Error saving splits to JSON file {split_file_path}: {e}")
        return None

    return splits_data


def get_dataloaders_from_splits(dataset, batch_size, fold_index, splits_data, dynamic_data_augmentation="F"):
    """
    Creates DataLoaders for a specific K-Fold split using pre-calculated indices.

    Args:
        dataset (MRIDataset): The full dataset.
        batch_size (int): Batch size for DataLoaders.
        fold_index (int): The current fold index (0 to kfold-1) to use for validation.
        splits_data (dict): The dictionary containing pre-calculated splits
                           (from generate_and_save_splits or loaded from JSON).
        dynamic_data_augmentation (str): String specifying dynamic augmentation.

    Returns:
        tuple: (train_loader, val_loader, test_loader, train_weights) for the specified fold.
               test_loader is None if no test indices are in splits_data.
               train_weights are computed for the current training fold.
        Returns None if inputs are invalid or loaders cannot be created.
    """
    if not splits_data or 'folds' not in splits_data or 'test_indices' not in splits_data:
        print("Error: Invalid or missing splits_data provided to get_dataloaders_from_splits.")
        return None
    if fold_index >= len(splits_data['folds']):
        print(f"Error: fold_index {fold_index} is out of bounds for the {len(splits_data['folds'])} folds in splits_data.")
        return None

    # --- 1. Retrieve Indices ---
    try:
        train_indices = splits_data['folds'][fold_index]['train']
        val_indices = splits_data['folds'][fold_index]['val']
        test_indices = splits_data['test_indices']

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

    except KeyError as e:
         print(f"Error: Malformed splits_data dictionary, missing key: {e}")
         return None
    except Exception as e:
         print(f"Error retrieving indices from splits_data: {e}")
         return None


    print(f"\n--- Creating DataLoaders for Fold {fold_index + 1}/{len(splits_data['folds'])} ---")
    print(f"Using pre-calculated indices: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # --- 2. Compute Class Weights for the Current Training Fold ---
    if not hasattr(dataset, 'outputs') or dataset.outputs is None:
         print("Error: Dataset outputs not loaded. Cannot compute class weights.")
         return None

    train_weights = dataset.compute_class_weights(train_indices)
    if train_weights is None or len(train_weights) == 0:
        print("Warning: Could not compute valid train weights. Training might be affected.")
        return None


    val_weights = dataset.compute_class_weights(val_indices)
    test_weights = dataset.compute_class_weights(test_indices)
    print(f"Train Class weights (Fold {fold_index + 1}): {train_weights}")
    print(f"Validation Class weights (Fold {fold_index + 1}): {val_weights}")
    print(f"Test Class weights (Fixed): {test_weights}")


    # --- 3. Create Subsets and DataLoaders ---
    original_transform = dataset.transform

    # Apply dynamic augmentation ONLY for the training set of this fold
    dynamic_transform_func = None
    if isinstance(dynamic_data_augmentation, str) and dynamic_data_augmentation.split("+")[0] == "T":
        try:
            dynamic_transform_func = get_dynamic_transforms(dynamic_data_augmentation)
        except Exception as e:
             print(f"Warning: Failed to create dynamic transforms from '{dynamic_data_augmentation}': {e}. No dynamic augmentation will be applied.")
             dynamic_transform_func = None

    # Assign transform only to the train subset temporarily
    dataset.transform = dynamic_transform_func

    try:
        # Check if indices are valid for the current dataset size
        if len(train_indices) > 0 and max(train_indices) >= len(dataset):
            raise IndexError(f"Max train index {max(train_indices)} out of bounds for dataset size {len(dataset)}")
        if len(val_indices) > 0 and max(val_indices) >= len(dataset):
            raise IndexError(f"Max validation index {max(val_indices)} out of bounds for dataset size {len(dataset)}")

        train_dataset = Subset(dataset, train_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    except IndexError as e:
         print(f"Error creating training Subset/DataLoader: {e}")
         dataset.transform = original_transform # Restore transform
         return None

    # IMPORTANT: Reset transform before creating validation/test loaders
    dataset.transform = None

    try:
        val_dataset = Subset(dataset, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    except IndexError as e:
         print(f"Error creating validation Subset/DataLoader: {e}")
         dataset.transform = original_transform # Restore transform
         return None

    test_loader = None
    if len(test_indices) > 0:
        try:
            if max(test_indices) >= len(dataset):
                 raise IndexError(f"Max test index {max(test_indices)} out of bounds for dataset size {len(dataset)}")
            test_dataset = Subset(dataset, test_indices)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print(f"Test set size: {len(test_dataset)}")
        except IndexError as e:
             print(f"Error creating test Subset/DataLoader: {e}")
             # Continue without test loader? Or return None? Let's continue without it.
             test_loader = None
             print("Proceeding without test loader due to index error.")
        except Exception as e:
             print(f"Unexpected error creating test loader: {e}")
             test_loader = None
    else:
        print("Test set size: 0 (no test indices provided)")

    # Restore original transform state in case the dataset object is reused elsewhere
    dataset.transform = original_transform

    print(f"Train loader size (batches): {len(train_loader)}")
    print(f"Validation loader size (batches): {len(val_loader)}")
    if test_loader:
        print(f"Test loader size (batches): {len(test_loader)}")

    # Return weights as tensor, handling None case
    train_weights_tensor = train_weights if train_weights is not None else torch.FloatTensor([])

    return train_loader, val_loader, test_loader, train_weights_tensor

