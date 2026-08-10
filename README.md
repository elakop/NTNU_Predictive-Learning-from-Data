# NTNU_Predictive-Learning-from-Data
This was a group project at National Taiwan Normal university where I was as an exchange student. I'm adding my part to github for completing my portfolio
Hybrid ML pipeline for predictive quality classification in Wire Arc Additive Manufacturing (WAAM), using audio and current/voltage sensor data
Predictive Quality Classification in Additive Manufacturing

A machine learning project for detecting defective welds (iO/niO) in Wire Arc Additive Manufacturing (WAAM) using multimodal sensor data — welding audio, current, and voltage signals. Statistical and spectral features were extracted from both signal types (~96 features, ~200 samples) and used to train supervised classifiers, including Random Forest and Logistic Regression with forward feature selection. Careful attention was paid to avoiding data leakage by restricting feature selection to the training split only. The goal was to enable real-time, non-destructive quality prediction as an alternative to post-process inspection.
