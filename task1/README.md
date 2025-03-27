## Overview
This script evaluates the performance of an object detection model (Yolov4) by analyzing precision, recall, and F1-score for different image sizes. It calculates the **AUC (Area Under the Curve)** of the **Precision-Recall curve** and determines the best confidence threshold based on the highest F1-score and best image size based on the auc.

## How It Works
1. **Reads** detection results from text files for different image sizes.
2. **Computes** precision, recall, and F1-score for various confidence thresholds.
3. **Calculates** the AUC of the Precision-Recall curve.
4. **Generates** Precision-Recall curves and saves them as images.
5. **Identifies** the best confidence threshold based on the highest F1-score.
5. **Saves** the best image size configuration in a text file.


## Justification for Selecting the Best Image Size
The best image size is chosen based on the **highest AUC value**. AUC represents the overall trade-off between precision and recall, making it a reliable metric for model performance evaluation. Additionally, the best confidence threshold is selected based on the **highest F1-score**, ensuring a balance between precision and recall.

## Run Remote 
- https://codesandbox.io/p/devbox/numpy-matplotlib-3dvmcn

## Run Local - Requirements
- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn


## Execution
Run the script using:
```bash
python main.py
```

## Output
- **Precision-Recall curve images** saved in `results/`.
- **Best configuration** saved in `results/best_configuration.txt` with the best image size and threshold information.


