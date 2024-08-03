import pytest
from src.visualization import plot_confusion_matrix

def test_plot_confusion_matrix(capsys):
    # Mock data
    y_test = [0, 1, 0, 1]
    y_pred = [0, 0, 1, 1]
    
    plot_confusion_matrix(y_test, y_pred)
    captured = capsys.readouterr()
    assert "Confusion matrix" in captured.out