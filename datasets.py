import os
import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, Subset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit


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
        subjects_id = []
        subjects_session = []
        subjects_amyloid = []

        for subject in subjects_input.keys():
            for ses in subjects_input[subject].keys():
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

                if group in self.target_variable and group in self.control_variable:
                    if self.amyloid == "C0T1" and amyloid_status == 0.:
                        outputs.append(0)
                    elif self.amyloid == "C0T1" and amyloid_status == 1.:
                        outputs.append(1)
                    elif self.amyloid == "C1T0" and amyloid_status == 0.:
                        outputs.append(1)
                    elif self.amyloid == "C1T0" and amyloid_status == 1.:
                        outputs.append(0)
                    else:
                        continue
                elif group in self.target_variable:
                    outputs.append(1)
                elif group in self.control_variable:
                    outputs.append(0)
                else:
                    continue

                # Stack the 3D inputs
                input_3d = np.stack([subject_map_1, subject_map_2, subject_map_3], axis=0)
                inputs_3d.append(input_3d)
                subjects_id.append(subject)
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
            raise ValueError("Input shape must be (110, 110, 68) or (116, 116, 81)")

        for i in range(len(inputs_3d)):
            for j in range(3):
                inputs_2d[i, j] = normalize_input(reshape_to_grid(inputs_3d[i, j]))

        self.inputs = torch.from_numpy(inputs_2d).float()
        self.outputs = torch.from_numpy(outputs)
        self.subjects_id = subjects_id
        self.subjects_session = subjects_session
        self.subjects_amyloid = subjects_amyloid

        class_counts = np.bincount(self.outputs.numpy().astype(int).flatten())
        total_samples = np.sum(class_counts)
        self.class_weights = torch.FloatTensor(total_samples / (len(class_counts) * class_counts))
        print(f"Class weights: {self.class_weights}")

        print(f"Loaded {len(self)} subjects")
        print(f"Input shape: {self.inputs.shape}")

        self.outputs = F.one_hot(self.outputs.to(torch.int64)).float()
        print(f"Output shape: {self.outputs.shape}")


    def get_sampler(self):
        sample_weights = self.class_weights[self.outputs.cpu().numpy().astype(int).flatten()]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def compute_class_weights(self, indices=None):
        if indices is None:
            indices = range(len(self.outputs))

        labels = torch.argmax(self.outputs[indices], dim=1).cpu().numpy()

        class_counts = np.bincount(labels.astype(int).flatten())
        total_samples = np.sum(class_counts)

        # Handle the case where a class might have zero samples
        class_weights = np.zeros_like(class_counts, dtype=np.float32)
        non_zero_counts = class_counts[class_counts > 0]
        class_weights[class_counts > 0] = total_samples / (len(non_zero_counts) * non_zero_counts)

        return torch.FloatTensor(class_weights)


def get_train_val_test_dataloaders(dataset, batch_size, val_ratio=0.15, test_ratio=0.15,
                                   seed=42, dynamic_data_augmentation="F"):

    # Extract subject IDs
    subject_ids = np.array([subject_id.split('_')[0] for subject_id in dataset.subjects_id])
    unique_subjects = np.unique(subject_ids)

    init_size = dataset.outputs.shape[0]
    print("Dataset init_size: ", init_size)

    # Create splits based on unique subjects
    test_size = int(test_ratio * len(unique_subjects))
    val_size = int(val_ratio * len(unique_subjects))

    # Split subjects into train+val and test
    if test_size == 0:
        train_val_idx = np.arange(len(unique_subjects))
        test_idx = []
    else:
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(X=unique_subjects, groups=unique_subjects))

    # Split the remaining subjects into train and val
    train_val_subjects = unique_subjects[train_val_idx]
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss_val.split(X=train_val_subjects, groups=train_val_subjects))

    # Get the indices for each split
    train_subjects = train_val_subjects[train_idx]
    val_subjects = train_val_subjects[val_idx]
    test_subjects = unique_subjects[test_idx]

    train_indices = np.where(np.isin(subject_ids, train_subjects))[0]
    val_indices = np.where(np.isin(subject_ids, val_subjects))[0]
    test_indices = np.where(np.isin(subject_ids, test_subjects))[0]

    # Compute class weights based on the combined train and validation set
    train_weights = dataset.compute_class_weights(train_indices)
    print(f"Train Class weights: {train_weights}")
    val_weights = dataset.compute_class_weights(val_indices)
    print(f"Validation Class weights: {val_weights}")
    test_weights = dataset.compute_class_weights(test_indices)
    print(f"Test Class weights: {test_weights}")
    print(f"Total class weights: {dataset.class_weights}")

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    if dynamic_data_augmentation.split("+")[0] == "T":
        dynamic_transforms = get_dynamic_transforms(dynamic_data_augmentation)
        train_dataset.dataset.transform = dynamic_transforms  # Apply dynamic augmentation only to the training set

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if len(test_indices) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
