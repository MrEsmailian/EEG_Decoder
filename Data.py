import os
import glob
import numpy as np
import mne
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset

class EEGDataLoader:
    def __init__(self, data_path, cache_path, batch_size=32, split_ratios=(0.7, 0.15, 0.15)):
        """
        Initializes the data loader factory for the 001-2017 dataset.
        """
        self.data_path = data_path
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.split_ratios = split_ratios
        
        self.event_mapping = {
            1536: 'Elbow Flexion',
            1537: 'Elbow Extension',
            1538: 'Supination',
            1539: 'Pronation',
            1540: 'Hand Open',
            1541: 'Hand Close',
            1542: 'Rest'
        }
        
        # MNE preprocessing parameters
        self.l_freq = 0.5
        self.h_freq = 40.0
        self.sfreq_new = 256  # [EDIT 3] Changed from 128 to 256
        self.tmin = 0.0       # [EDIT 2] Changed from -1.0 to 0.0
        self.tmax = 3.0       
        
        os.makedirs(self.cache_path, exist_ok=True)

    def get_class_names(self):
        return list(self.event_mapping.values())

    def _get_cache_filename(self, subject):
        return os.path.join(self.cache_path, f"preprocessed_sub_{subject}.pt")

    def _process_subject_run(self, file_path):
        """Reads a single GDF file, filters, epochs, and standardizes it."""
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose='ERROR')
        # Keep only the first 61 channels (drops EOG and kinematic sensors)
        raw.pick(raw.ch_names[:61])
        raw.apply_function(lambda x: np.nan_to_num(x, copy=False))
        iir_params = dict(order=4, ftype='butter', output='sos')
        raw.filter(self.l_freq, self.h_freq, method='iir', iir_params=iir_params, phase='zero', verbose='ERROR')
        raw.notch_filter(50.0, verbose='ERROR')
        raw.resample(self.sfreq_new, npad='auto')
        raw.set_eeg_reference(ref_channels='average', projection=False, verbose='ERROR')
        
        events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
        target_events = []
        event_code_to_label_idx = {} 
        
        for event in events:
            annotation_desc = list(event_id.keys())[list(event_id.values()).index(event[2])]
            try:
                code = int(annotation_desc)
                if code in self.event_mapping:
                    target_events.append(event)
                    event_code_to_label_idx[event[2]] = list(self.event_mapping.keys()).index(code)
            except ValueError:
                continue 

        if len(target_events) == 0:
            return None, None
        target_events = np.array(target_events)
        epochs = mne.Epochs(
            raw, target_events, tmin=self.tmin, tmax=self.tmax, 
            baseline=None, preload=True, verbose='ERROR'
        )
        
        if len(epochs) == 0:
            return None, None
        data = epochs.get_data(copy=True)
        final_labels = [event_code_to_label_idx[e_code] for e_code in epochs.events[:, 2]]
        
        return data, np.array(final_labels)

    def _load_and_preprocess_subject(self, subject):
        """Processes all runs for a SINGLE subject and saves to cache."""
        all_x = []
        all_y = []
        
        print(f"Preprocessing data for Subject {subject}...")
        for run in range(1, 11):
            file_pattern = os.path.join(self.data_path, f"*_subject{subject}_run{run}*.gdf")
            files = glob.glob(file_pattern)
            
            for f in files:
                x, y = self._process_subject_run(f)
                if x is not None:
                    all_x.append(x)
                    all_y.append(y)
                    
        if not all_x:
            raise FileNotFoundError(f"No valid GDF files found for Subject {subject}.")

        X_np = np.concatenate(all_x, axis=0)
        Y_np = np.concatenate(all_y, axis=0)
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1) 
        Y_tensor = torch.tensor(Y_np, dtype=torch.long)
        
        cache_file = self._get_cache_filename(subject)
        torch.save({'X': X_tensor, 'Y': Y_tensor}, cache_file)
        print(f"Cached Subject {subject} to {cache_file}")
        
        return X_tensor, Y_tensor

    def get_dataloaders(self, subject, random_seed=42):
        """
        Takes a specific subject ID, loads their preprocessed data (from cache if available), 
        splits them, NORMALIZES based ONLY on train data to prevent data leakage,
        and returns Train, Validation, and Test DataLoaders.
        """
        cache_file = self._get_cache_filename(subject)
        
        if os.path.exists(cache_file):
            print(f"Loading Subject {subject} from cache...")
            data = torch.load(cache_file)
            X_tensor, Y_tensor = data['X'], data['Y']
        else:
            X_tensor, Y_tensor = self._load_and_preprocess_subject(subject)

        total_size = len(X_tensor)
        train_size = int(self.split_ratios[0] * total_size)
        val_size = int(self.split_ratios[1] * total_size)
        test_size = total_size - train_size - val_size 
        
        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(total_size, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        X_train = X_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        
        X_val = X_tensor[val_indices]
        Y_val = Y_tensor[val_indices]
        
        X_test = X_tensor[test_indices]
        Y_test = Y_tensor[test_indices]
        
        print("Normalizing data based on training set statistics...")
        mean = X_train.mean(dim=(0, 3), keepdim=True)
        std = X_train.std(dim=(0, 3), keepdim=True)
        
        X_train = (X_train - mean) / (std + 1e-8)
        
        if len(X_val) > 0:
            X_val = (X_val - mean) / (std + 1e-8)
        if len(X_test) > 0:
            X_test = (X_test - mean) / (std + 1e-8)
        
        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def get_k_fold_dataloaders(self, subject, k=5, random_seed=42):
        """
        Creates K-Fold Cross-Validation DataLoaders for a specific subject.

        Returns:
            List of k tuples:
            [(train_loader, val_loader), ...]
        """
        if k < 2:
            raise ValueError("k must be at least 2.")

        cache_file = self._get_cache_filename(subject)

        if os.path.exists(cache_file):
            print(f"Loading Subject {subject} from cache...")
            data = torch.load(cache_file)
            X_tensor, Y_tensor = data['X'], data['Y']
        else:
            X_tensor, Y_tensor = self._load_and_preprocess_subject(subject)

        total_size = len(X_tensor)

        if k > total_size:
            raise ValueError(
                f"k={k} cannot be greater than the number of samples ({total_size})."
            )

        generator = torch.Generator().manual_seed(random_seed)
        indices = torch.randperm(total_size, generator=generator).tolist()

        base_fold_size = total_size // k
        remainder = total_size % k
        fold_sizes = [
            base_fold_size + (1 if i < remainder else 0)
            for i in range(k)
        ]

        folds = []
        current = 0

        for fold_size in fold_sizes:
            folds.append(indices[current:current + fold_size])
            current += fold_size

        fold_loaders = []

        for fold_idx in range(k):
            print(f"\n{'=' * 20} Fold {fold_idx + 1}/{k} {'=' * 20}")

            val_indices = folds[fold_idx]
            train_indices = []

            for i in range(k):
                if i != fold_idx:
                    train_indices.extend(folds[i])

            X_train = X_tensor[train_indices]
            Y_train = Y_tensor[train_indices]
            X_val = X_tensor[val_indices]
            Y_val = Y_tensor[val_indices]

            print("Normalizing using training-set statistics...")

            mean = X_train.mean(dim=(0, 3), keepdim=True)
            std = X_train.std(dim=(0, 3), keepdim=True)

            X_train = (X_train - mean) / (std + 1e-8)
            X_val = (X_val - mean) / (std + 1e-8)

            train_dataset = TensorDataset(X_train, Y_train)
            val_dataset = TensorDataset(X_val, Y_val)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

            fold_loaders.append((train_loader, val_loader))

        return fold_loaders

    def combined_dataloaders(self, subjects, batch_size=32, shuffle=True, random_seed=42, train=True):
        """
        Combine multiple DataLoaders into one DataLoader.

        Samples are shuffled individually, not batches.

        Args:
            dataloaders: list of PyTorch DataLoaders
            batch_size: batch size of the resulting DataLoader
            shuffle: whether to shuffle samples
            num_workers: number of DataLoader workers

        Returns:
            A single DataLoader containing all samples.
        """
        dataloaders = []
        for s in subjects:
            trainloader, valloader, _ = self.get_dataloaders(s, random_seed)
            if train:
                dataloaders.append(trainloader)
            else:
                dataloaders.append(valloader)

        datasets = [loader.dataset for loader in dataloaders]

        combined_dataset = ConcatDataset(datasets)
        generator = torch.Generator().manual_seed(random_seed)
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            pin_memory=True
        )

        return combined_loader