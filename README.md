# Informer network fusing interpretability and dynamic frequency denoising without information leakage for predicting complex systems

This repository contains the experiment code for the Informer-based time-series prediction project associated with the paper below. Dataset files, trained weights, and generated results are excluded.

## Paper

Shijian Dong, Tianyu Yu, Lixin Han, and Jianguo Dong, "Informer network fusing interpretability and dynamic frequency denoising without information leakage for predicting complex systems," *IEEE Sensors Journal*, DOI: [10.1109/JSEN.2025.3615981](https://doi.org/10.1109/JSEN.2025.3615981).

Use the DOI page to access the paper. The publisher PDF is not copied into this repository.

## Dataset and training

Provide your own numeric CSV dataset outside version control. Every column except the final column is treated as an input feature; the final column is the target. The script scales features and targets using training rows only. It uses a chronological split of 3,510 training windows, 390 validation windows, and 100 held-out test windows with a 256-step input window, so the CSV must contain at least 4,256 rows. The checkpoint is selected by validation loss, and the test split is evaluated once afterward.

Run from this directory:

```bash
python main_inf_MS_VMD_contrast.py --data /path/to/your_dataset.csv
```

Install `numpy`, `pandas`, `scikit-learn`, `torch`, `matplotlib`, and `scipy` first. Optional generated output paths can be set with `--checkpoint` and `--results-dir`; generated checkpoints, results, and scaler parameters are ignored by Git.

The optional ElasticNet notebook expects 26 numeric input columns followed by one target column. MATLAB scripts under `data/` and `run_result_MS/process/` may require external datasets and MATLAB toolboxes.

