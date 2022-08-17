# Reliability Diagram

Reliability diagram shows the accuracy of prediction versus the confidence of a model. The code,a also calculates the Expected Calibration Error (ECE).

## Dependancy:
PyTorch (only if the raw inputs are given)

## How to run:
The code can be executed with the prediction probabilities, or with raw outputs of the model. The difference is that model outputs are numerical values (could be anything, even a negative number), while probabilities are positive numbers and must sum to 1 for each prediction. If raw outputs are provided, the code uses a softmax layer to convert the outputs to probabilities.

### Run with the probabilities
```
from reliability_diagram import plot_reliability_diagram
plot_reliability_diagram reliability_diagram(y_true, y_pred, probs=probs, nbins=20)
```

### Run with the model output 
```
from reliability_diagram import plot_reliability_diagram
reliability_diagram(y_true, y_pred, raw_output=raw_output, nbins=20)
```
