# tests/test_model.py
import pytest
import torch
from model import ImageClassifier

def test_model_output():
    model = ImageClassifier(num_classes=6)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    
    assert output.shape == (1, 6), "Output shape should be (batch_size, num_classes)"
    
def test_model_training_step():
    model = ImageClassifier(num_classes=6)
    batch = (
        torch.randn(4, 3, 224, 224),  # Images
        torch.randint(0, 6, (4,))      # Labels
    )
    loss = model.training_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor), "Training step should return a loss tensor"
    assert loss.requires_grad, "Loss should require gradients"