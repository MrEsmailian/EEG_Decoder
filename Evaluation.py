import os
import torch
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class visualizer:

    @staticmethod
    def plot_learning_curves(train_losses, val_losses, title_exp='Loss'):
        """
        Plots the training and validation loss curves over epochs.
        Args:
            train_losses (list of floats): Loss values for the training set per epoch.
            val_losses (list of floats): Loss values for the validation set per epoch.
            save_path (str, optional): If provided, saves the plot to this file path.
        """
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title(f'Training and Validation {title_exp}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        # Force x-axis to use integer ticks for epochs
        plt.xticks(epochs) 
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(model, data_loader, class_names, title='Confusion Matrix'):
        model.model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(model.device)
                y = y.to(model.device)
                outputs = model.model(X)
                predictions = torch.argmax(outputs, dim=1)
                all_targets.extend(y.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        cm = confusion_matrix(all_targets, all_predictions, labels=np.arange(len(class_names)))
        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.show()
        return cm

    @staticmethod
    def plot_model_performance(labels, test_means, test_vars, val_means, val_vars):
        test_std = np.sqrt(np.array(test_vars))
        val_std = np.sqrt(np.array(val_vars))
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, test_means, width, yerr=test_std, capsize=5, label="Test")
        ax.bar(x + width / 2, val_means, width, yerr=val_std, capsize=5, label="Validation")
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()

        ax.grid(axis="y", alpha=0.3)
        all_values = np.concatenate([
            np.array(test_means) - test_std,
            np.array(test_means) + test_std,
            np.array(val_means) - val_std,
            np.array(val_means) + val_std
        ])

        y_min = all_values.min()
        y_max = all_values.max()
        margin = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - margin, y_max + margin)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_one_sample_per_label(data_loader_factory, subject):
        cache_file = data_loader_factory._get_cache_filename(subject)
        if not os.path.exists(cache_file):
            print(f"Cache for Subject {subject} does not exist.")
            print("Run get_dataloaders(subject) first to preprocess the data.")
            return

        data = torch.load(cache_file)
        X = data['X']
        Y = data['Y']
        X = X.squeeze(1).numpy()
        Y = Y.numpy()
        class_names = data_loader_factory.get_class_names()

        # ---------------------------------------------------------
        # Find one sample from each class
        # ---------------------------------------------------------
        samples = {}
        for class_idx, class_name in enumerate(class_names):
            indices = np.where(Y == class_idx)[0]
            if len(indices) == 0:
                print(f"No sample found for class: {class_name}")
                continue
            sample_idx = indices[0]
            samples[class_idx] = X[sample_idx]

        # ---------------------------------------------------------
        # Plot
        # ---------------------------------------------------------
        n_classes = len(samples)
        fig, axes = plt.subplots(n_classes, 1, figsize=(15, 3 * n_classes), sharex=True)
        if n_classes == 1:
            axes = [axes]
        for ax, (class_idx, sample) in zip(axes, samples.items()):
            n_channels, n_times = sample.shape
            time = np.arange(n_times) / data_loader_factory.sfreq_new
            for ch in range(n_channels):
                ax.plot(time, sample[ch] + ch * 5, linewidth=0.5)
            ax.set_ylabel("Channels")
            ax.set_title(f"Class {class_idx}: {class_names[class_idx]}")
            ax.grid(True, alpha=0.2)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"One EEG Sample per Class - Subject {subject}", fontsize=16)
        plt.tight_layout()
        plt.show()

class ExperimentDataSaver:
    def __init__(self, root_path):
        self.root = root_path
        self.data = {}
        os.makedirs(self.root, exist_ok=True)

    def set_data(self, model, phase, stats, subjects, values):
        models = [model] if isinstance(model, str) else model
        phases = [phase] if isinstance(phase, str) else phase
        stats_list = [stats] if isinstance(stats, str) else stats
        subjects_list = [subjects] if isinstance(subjects, (int, str)) else subjects
        
        values_flat = np.atleast_1d(values).flatten()
        
        grid = list(itertools.product(models, phases, stats_list, subjects_list))
        
        for (m, p, s, sub), v in zip(grid, values_flat):
            sub_str = f"#{sub}" if isinstance(sub, int) or (isinstance(sub, str) and str(sub).isdigit()) else sub
            self.data[(m, p, s, sub_str)] = v

    def save_excel(self, file_name, verbose=True):
        file_path = os.path.join(self.root, file_name)
        
        if not self.data:
            if verbose:
                print("No data available to save.")
            return

        rows = [{'Model': k[0], 'Phase': k[1], 'Stats': k[2], 'Subjects': k[3], 'Value': v} 
                for k, v in self.data.items()]
        df = pd.DataFrame(rows)
        
        pivot_df = df.pivot_table(
            index=['Model', 'Phase', 'Stats'], 
            columns='Subjects', 
            values='Value', 
            aggfunc='first'
        )
        
        def extract_num(col):
            try:
                return int(str(col).replace('#', ''))
            except ValueError:
                return float('inf')
                
        sorted_cols = sorted(pivot_df.columns, key=extract_num)
        pivot_df = pivot_df[sorted_cols]
        
        pivot_df = pivot_df.where(pd.notnull(pivot_df), None)
        
        mode = 'a' if os.path.exists(file_path) else 'w'
        
        try:
            with pd.ExcelWriter(
                file_path, 
                engine='openpyxl', 
                mode=mode, 
                if_sheet_exists='new' if mode == 'a' else None
            ) as writer:
                sheet_name = datetime.now().strftime("Run_%H-%M-%S")
                pivot_df.to_excel(writer, sheet_name=sheet_name)
                
                if verbose:
                    print(f"Data saved successfully to {file_path} (Sheet: {sheet_name})")
        except Exception as e:
            if verbose:
                print(f"Failed to save Excel file: {e}")
        
        self.data = {}