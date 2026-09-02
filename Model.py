import torch
import torch.nn as nn
from torchsummary import summary

# ==========================================
# Fitting Policy
# ==========================================

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        # First epoch
        if self.best_loss is None:
            self.best_loss = val_loss
            return True # Indicates this is the best model so far
            
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
            
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True

# ==========================================
# Utility Layer
# ==========================================

class Expression(nn.Module):
    """Utility layer to allow custom lambda functions inside nn.Sequential."""
    def __init__(self, expr_fn):
        super().__init__()
        self.expr_fn = expr_fn
    def forward(self, x):
        return self.expr_fn(x)

# ==========================================
# Main Model Class
# ==========================================

class EEGModel:
    def __init__(self, criterion, learning_rate, architecture_key, random_seed, device):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        self.num_classes = 7
        
        self.architecture_key = architecture_key
        
        if architecture_key == 'DeepConvNet':
            self.model = self._build_DeepConvNet().to(self.device)
        elif architecture_key == 'ShallowFBCSPNet':
            self.model = self._build_ShallowFBCSPNet().to(self.device)
        elif architecture_key == 'CTNet':
            self.model = self._build_CTNet().to(self.device)
        elif architecture_key == 'Deep4Net':
            self.model = self._build_Deep4Net().to(self.device)
        elif architecture_key == 'EEGConformer': 
            self.model = self._build_EEGConformer().to(self.device)
        else:
            raise ValueError(f"Invalid architecture! Choose from: {list(self.models_dict.keys())}")
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # ==========================================
    # Architecture Definitions
    # ==========================================

    def _build_DeepConvNet(self):
        """DeepConvNet optimized for raw time-series feature extraction."""
        return nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), bias=False),
            nn.Conv2d(25, 25, (61, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            
            nn.Conv2d(25, 50, (1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            
            nn.Conv2d(50, 100, (1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            
            nn.Conv2d(100, 200, (1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 3)),
            nn.Dropout(0.5),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(1000, self.num_classes)
        )

    def _build_ShallowFBCSPNet(self):
        """Shallow CNN designed to mimic Filter Bank Common Spatial Patterns."""
        return nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), bias=False),
            nn.Conv2d(40, 40, (61, 1), bias=False),
            nn.BatchNorm2d(40),
            Expression(lambda x: x ** 2), # Square activation
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            Expression(lambda x: torch.log(torch.clamp(x, min=1e-6))), # Log activation (clamped)
            nn.Dropout(0.5),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(1800, self.num_classes)
        )
    
    def _build_CTNet(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 25), padding='same', bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(61, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), groups=32, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.25),
            nn.Flatten(),

            nn.Linear(32 * 48, self.num_classes)
        )
    
    def _build_Deep4Net(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 10), bias=False),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(61, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 10), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=200, out_channels=7, kernel_size=(1, 5), bias=True),
            nn.Flatten(),
            nn.Linear(7, self.num_classes)
        )
    
    # ==========================================
    # Core Functions
    # ==========================================

    def forward(self, x):
        return self.model(x)

    def train_one_epoch(self, train_loader, val_loader):
        self.model.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        self.model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        return epoch_train_loss, epoch_val_loss

    def evaluate_accuracy(self, dataloader):
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        accuracy = (correct_predictions / total_samples) * 100.0
        return accuracy

    def save_model(self, file_path, verbose=True):
        torch.save(self.model.state_dict(), file_path)
        if verbose:
            print(f"Model saved successfully to {file_path}")

    def load_model(self, file_path, verbose=True):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        if verbose:
            print(f"Model loaded successfully from {file_path}")

    def fit(self, train_loader, val_loader, max_epochs=50, patience=5, save_path="best_model.pth", verbose=True):
        early_stopping = EarlyStopping(patience=patience)
        
        for epoch in range(max_epochs):
            train_loss, val_loss = self.train_one_epoch(train_loader, val_loader)
            if verbose:
                print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            is_best = early_stopping(val_loss)
            if is_best:
                self.save_model(save_path, verbose=verbose)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
        if verbose:
            print("Training complete. Loading best model weights...")
        self.load_model(save_path, verbose=verbose)

    def freeze(self, freeze_classifier=True):
        for param in self.model.parameters():
            param.requires_grad = False
        if not freeze_classifier:
            if isinstance(self.model[-1], nn.Linear):
                for param in self.model[-1].parameters():
                    param.requires_grad = True
        print(f"Model frozen. Classifier frozen: {freeze_classifier}")

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        print("Model unfrozen. All parameters are now trainable.")
    
    # ==========================================
    # Getters and Setters
    # ==========================================

    def set_device(self, new_device):
        self.device = torch.device(new_device)
        self.model.to(self.device)

    def get_device(self):
        return self.device

    def set_learning_rate(self, new_lr):
        self.learning_rate = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def get_learning_rate(self):
        return self.learning_rate
    
    # ==========================================
    # Model Summary
    # ==========================================

    def model_summary(self):
        device_str = str(self.device).split(':')[0] 
        # Corrected for raw time-series input shape (1 Channel, 61 EEG Electrodes, 769 Time Steps)
        summary(self.model, input_size=(1, 61, 769), device=device_str)

# ==========================================
# Average Voting Ensemble Model Class
# ==========================================

class AverageEnsemble(nn.Module):
    def __init__(self, models):
        super(AverageEnsemble, self).__init__()
        extracted_models = [m.model if hasattr(m, 'model') else m for m in models]
        self.models = nn.ModuleList(extracted_models)
        
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        outputs_stacked = torch.stack(outputs)
        avg_output = torch.mean(outputs_stacked, dim=0)
        
        return avg_output
    
    def evaluate_accuracy(self, dataloader, device=None):
        self.eval()
        correct = 0
        total = 0
        
        # Automatically detect the device from the model weights
        if device is None:
            device = next(self.models[0].parameters()).device
            
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

# ==========================================
# Linea Classifier Ensemble Model Class
# ==========================================

class LinearEnsemble(nn.Module):
    def __init__(self, models_list, criterion, learning_rate, random_seed, device, freeze_models=True):
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        super().__init__()
            
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        self.num_classes = models_list[0].num_classes if len(models_list) > 0 else 7
        self.models = nn.ModuleList([m.model for m in models_list])
        self.classifier = nn.Linear(len(self.models) * self.num_classes, self.num_classes)
        
        self.to(self.device)
        
        if freeze_models:
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False
            print("Base models frozen. Only the ensemble linear classifier will be trained.")
        else:
            print("Base models unfrozen. The entire ensemble pipeline will be fine-tuned.")
            
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)

    # ==========================================
    # Core Functions
    # ==========================================

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        concatenated = torch.cat(outputs, dim=1)
        return self.classifier(concatenated)

    def train_one_epoch(self, train_loader, val_loader):
        self.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        self.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        return epoch_train_loss, epoch_val_loss

    def evaluate_accuracy(self, dataloader):
        self.eval()
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        accuracy = (correct_predictions / total_samples) * 100.0
        return accuracy

    def save_model(self, file_path, verbose=True):
        # We use self.state_dict() directly since the class is an nn.Module
        torch.save(self.state_dict(), file_path)
        if verbose:
            print(f"Ensemble model saved successfully to {file_path}")

    def load_model(self, file_path, verbose=True):
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        if verbose:
            print(f"Ensemble model loaded successfully from {file_path}")

    def fit(self, train_loader, val_loader, max_epochs=50, patience=5, save_path="best_ensemble.pth", verbose=True):
        early_stopping = EarlyStopping(patience=patience)
        
        for epoch in range(max_epochs):
            train_loss, val_loss = self.train_one_epoch(train_loader, val_loader)
            if verbose:
                print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            is_best = early_stopping(val_loss)
            if is_best:
                self.save_model(save_path, verbose=verbose)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                break
                
        if verbose:
            print("Training complete. Loading best model weights...")
        self.load_model(save_path, verbose=verbose)