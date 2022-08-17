# Reliability Diagram

Reliability diagram shows the accuracy of prediction versus the confidence of a model.

## Dependancy:
PyTorch (only if the raw inputs are provided, not the probabilities)

## How to run:
The code can be executed with the prediction probabilities, or with raw inputs of the model.

### Run with the probabilities
from reliability_diagram import plot_reliability_diagram

plot_reliability_diagram reliability_diagram(y_true, y_pred, probs=probs, nbins=20)

### Run with the model output 
from reliability_diagram import plot_reliability_diagram

reliability_diagram(y_true, y_pred, raw_output=raw_output, nbins=20)
