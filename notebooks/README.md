# Notebooks Guide
This folder contains all experiment notebooks. Each notebook isolates one idea, but together they show the progression of experiments.

---

## 📂 Index

**01_baseline.ipynb**  
- Purpose: Establish a reference accuracy with standard augmentations, but no dropout or cutout.  
- Key Findings: Validation accuracy of 82.4% (test 81.8%).  

**02_fc_dropout.ipynb**  
- Purpose: Add dropout only to fully connected layers.  
- Key Findings: Validation accuracy improved to 83.8% (test 83.7%).  

**03_conv_dropout.ipynb**  
- Purpose: Add dropout only to convolutional layers.  
- Key Findings: Validation accuracy reached 85.0% (test 83.3%). Strongest regularization overall.  

**04_cutout.ipynb**  
- Purpose: Apply cutout augmentation to the baseline CNN.  
- Key Findings: Validation accuracy 84.0% (test 84.0%). Best augmentation method.  

---

## 🔎 Notes
- Each notebook includes result tables comparing against the baseline.  
- Multiple dropout probabilities and cutout sizes were tested; only the best-performing settings are reported here.  
- **Comparison:** Dropout in convolutional layers gave the highest validation accuracy. Cutout matched dropout in test accuracy, showing its strength as a data augmentation technique.
