import mne
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

class EEGDatasetLoader:
    """
    A class to load, preprocess, and split the Upper Limb Movement EEG dataset.
    """
    def __init__(self, file_path, train_pct=0.7, val_pct=0.15, test_pct=0.15, random_seed=42):
        """
        Initializes the dataset loader, performs preprocessing, and extracts epochs.
        
        Args:
            file_path (str): Path to the 88MB GDF dataset file.
            train_pct (float): Percentage of data for the training set.
            val_pct (float): Percentage of data for the validation set.
            test_pct (float): Percentage of data for the test set.
            random_seed (int): Seed for shuffling the train set.
        """
        assert abs((train_pct + val_pct + test_pct) - 1.0) < 1e-5, "Percentages must sum to 1.0"
        
        self.file_path = file_path
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.random_seed = random_seed
        
        # Event codes defined in the paradigm documentation[cite: 2]
        self.event_mapping = {
            '1536': 0,  # 0x600: elbow flexion[cite: 2]
            '1537': 1,  # 0x601: elbow extension[cite: 2]
            '1538': 2,  # 0x602: supination[cite: 2]
            '1539': 3,  # 0x603: pronation[cite: 2]
            '1540': 4,  # 0x604: hand close[cite: 2]
            '1541': 5,  # 0x605: hand open[cite: 2]
            '1542': 6   # 0x606: rest[cite: 2]
        }
        
        self.X = None
        self.y = None
        
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """
        Internal method to handle the GDF loading and apply source-specified filters.
        """
        # Load GDF file[cite: 1, 2]
        raw = mne.io.read_raw_gdf(self.file_path, preload=True)
        
        # The EEG was measured from 61 channels (indices 0-60); other channels are EOG, glove, and exoskeleton data[cite: 1, 2]
        eeg_ch_names = raw.ch_names[:61] 
        raw.pick_channels(eeg_ch_names)
        
        # Ensure the data is sampled at 512 Hz[cite: 1, 2]
        if raw.info['sfreq'] != 512.0:
            raw.resample(512.0)
        
        # Apply an 8th order Chebyshev bandpass filter from 0.01 Hz to 200 Hz[cite: 1, 2]
        # We use Chebyshev Type I ('cheby1') specifying the passband ripple parameter (rp).
        iir_params = dict(order=8, ftype='cheby1', rp=0.5)
        raw.filter(l_freq=0.01, h_freq=200.0, method='iir', iir_params=iir_params)
        
        # Power line interference was suppressed with a notch filter at 50 Hz[cite: 1, 2]
        raw.notch_filter(freqs=50.0)
        
        # Re-reference the data to a common average reference (CAR)[cite: 1]
        raw.set_eeg_reference('average')
        
        # Extract events directly from the GDF annotations
        events, event_dict_mne = mne.events_from_annotations(raw)
        
        # Filter for valid target movement/rest events and map them to our internal 0-6 class labels
        valid_events = []
        for event in events:
            event_id_mne = event[2]
            # MNE translates string annotations to integers; we reverse-map to check against our hex-to-dec codes
            event_desc = list(event_dict_mne.keys())[list(event_dict_mne.values()).index(event_id_mne)]
            
            if event_desc in self.event_mapping:
                # Replace MNE's arbitrary ID with our continuous 0-6 class IDs
                event[2] = self.event_mapping[event_desc]
                valid_events.append(event)
                
        valid_events = np.array(valid_events)
        
        # Extract epochs: we extract data from 0s to 3s relative to the cue onset.
        # At second 2, a cue was presented on the computer screen[cite: 1, 2]
        epochs = mne.Epochs(raw, valid_events, event_id=None, tmin=0.0, tmax=3.0, 
                            baseline=None, preload=True)
        
        # X shape: (trials, channels, timepoints), y shape: (trials,)
        self.X = epochs.get_data()
        self.y = epochs.events[:, 2] 

    def get_unique_classes(self):
        """
        Returns the unique class labels present in the loaded dataset.
        
        Returns:
            numpy.ndarray: Array of unique integer labels.
        """
        if self.y is None:
            raise ValueError("Dataset has not been loaded correctly.")
        return np.unique(self.y)

    def get_dataloaders(self, batch_size):
        """
        Splits the dataset and returns PyTorch DataLoaders.
        
        Args:
            batch_size (int): The batch size for the DataLoaders.
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset has not been loaded correctly.")
            
        # Convert to PyTorch tensors
        tensor_X = torch.tensor(self.X, dtype=torch.float32)
        tensor_y = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(tensor_X, tensor_y)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(self.train_pct * total_size)
        val_size = int(self.val_pct * total_size)
        test_size = total_size - train_size - val_size
        
        # Split using the provided random seed for deterministic shuffling
        generator = torch.Generator().manual_seed(self.random_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Create DataLoaders (only train loader is shuffled per epoch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader