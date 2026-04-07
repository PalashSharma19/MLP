# Deferred Tasks Tracker

This file tracks tasks intentionally deferred for later (hardware/time dependent).

## Pending

1. Run full CNN training on real dataset
- Command: C:/Users/gamin/miniconda3/python.exe src/train_cnn.py
- Expected output: outputs/models/resnet18_phase1.pth and outputs/models/resnet18_best.pth
- Notes: GPU recommended; long run.

2. Run full ML training on real dataset
- Command: C:/Users/gamin/miniconda3/python.exe src/train_ml.py
- Expected output: outputs/models/rf_model.pkl, outputs/models/svm_model.pkl, outputs/models/knn_model.pkl, outputs/models/ml_results.json
- Notes: SVM may take several minutes.

3. Run full unified evaluation after all models are trained
- Command: C:/Users/gamin/miniconda3/python.exe src/evaluate.py
- Expected output: outputs/results.csv
- Notes: Requires CNN and ML model artifacts.

4. Re-run Section 9 visualizations after full training and evaluation
- Command: C:/Users/gamin/miniconda3/python.exe src/visualize.py
- Expected output: outputs/plots/cm_CNN.png, cm_RF.png, cm_SVM.png, cm_kNN.png, model_comparison.png, training_curves.png, sample_images.png
- Notes: Right now only sample_images.png can be generated because trained model outputs are not present yet.

5. Run full pipeline end-to-end once on charger/internet
- Command: C:/Users/gamin/miniconda3/python.exe main.py
- Expected output: full model artifacts, outputs/results.csv, and populated outputs/plots/
- Notes: This is the full heavy run.

6. Re-run evaluation+plots only after training artifacts exist
- Command: C:/Users/gamin/miniconda3/python.exe main.py --skip-convert --skip-cnn --skip-ml
- Expected output: refreshed outputs/results.csv and outputs/plots/*

7. Re-run just plots from existing artifacts
- Command: C:/Users/gamin/miniconda3/python.exe main.py --skip-convert --skip-cnn --skip-ml --skip-features
- Expected output: refreshed outputs/plots/*

8. Validate inference on a real sample file (CNN)
- Command: C:/Users/gamin/miniconda3/python.exe predict.py --file /path/to/some.exe --model cnn
- Notes: Requires outputs/models/resnet18_best.pth from full CNN training.

9. Validate inference on a real sample file (RF)
- Command: C:/Users/gamin/miniconda3/python.exe predict.py --file /path/to/some.exe --model rf
- Notes: Requires outputs/models/rf_model.pkl from full ML training.
