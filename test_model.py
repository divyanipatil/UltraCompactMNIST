import torch
from torchvision import transforms
from train import UltraCompactCNN, count_parameters


model_path = 'models/mnist_model.pth'
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def test_total_parameter_count():
    model = UltraCompactCNN()
    param_count = count_parameters(model)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"


def test_batch_normalization():
    model = UltraCompactCNN()
    batch_norm_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
    assert len(batch_norm_layers) > 0, "Model should use Batch Normalization"


def test_dropout():
    model = UltraCompactCNN()
    dropout_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Dropout2d)]
    assert len(dropout_layers) > 0, "Model should use DropOut"


def test_fully_connected_or_gap():
    model = UltraCompactCNN()
    has_fully_connected = any(isinstance(layer, torch.nn.Linear) for layer in model.modules())
    has_gap = any(isinstance(layer, torch.nn.AdaptiveAvgPool2d) for layer in model.modules())
    assert has_fully_connected or has_gap, "Model should have a Fully Connected Layer or Global Average Pooling (GAP)"