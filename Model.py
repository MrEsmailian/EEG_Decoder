import torch
import torch.nn as nn
from torchsummary import summary

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
        
        self.models_dict = {
            'EEGNet': self._build_EEGNet(),
            'DeepConvNet': self._build_DeepConvNet(),
            'ShallowFBCSPNet': self._build_ShallowFBCSPNet()
        }
        
        if architecture_key not in self.models_dict:
            raise ValueError(f"Invalid architecture! Choose from: {list(self.models_dict.keys())}")
            
        self.model = self.models_dict[architecture_key].to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # ==========================================
    # Architecture Definitions
    # ==========================================
    
    def _build_EEGNet(self):
        """Standard EEGNet adapted for (1, 61, 769) raw time-series input."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (61, 1), groups=8, bias=False), # Spatial Filter
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(16, 16, (1, 16), groups=16, padding='same', bias=False), # Depthwise
            nn.Conv2d(16, 16, (1, 1), bias=False), # Pointwise
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(384, self.num_classes)
        )

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

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved successfully to {file_path}")

    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        print(f"Model loaded successfully from {file_path}")

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