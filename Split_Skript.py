"""
Data Split Script - 80/20 Train/Test
=====================================
Rozdělí data na train (80%) a test (20%) a uloží do CSV souborů.
Použij tyto soubory pro všechny modely (RF, Logistic Regression, atd.)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "C:/Users/Administrator/Documents/MUNI/2/NTNU_fall/Predictive Learning from Data/group_project/Data/All_Features_New.csv"
OUTPUT_TRAIN = "train_80.csv"
OUTPUT_TEST = "test_20.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 60)
print("DATA SPLIT: 80% TRAIN / 20% TEST")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded: {len(df)} samples, {len(df.columns)} columns")

# Create class labels
def create_class_label(row):
    if row['Label'] == 'iO':
        return 'ok'
    else:
        return row['Error_Description'].lower()

df['Class'] = df.apply(create_class_label, axis=1)

print(f"\nOriginal class distribution:")
print(df['Class'].value_counts())

# ============================================================================
# STRATIFIED SPLIT
# ============================================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df['Class'],
    random_state=RANDOM_STATE
)

print(f"\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print(f"\nTrain set: {len(train_df)} samples ({100*(1-TEST_SIZE):.0f}%)")
print(train_df['Class'].value_counts())

print(f"\nTest set: {len(test_df)} samples ({100*TEST_SIZE:.0f}%)")
print(test_df['Class'].value_counts())

# ============================================================================
# SAVE TO CSV
# ============================================================================

train_df.to_csv(OUTPUT_TRAIN, index=False)
test_df.to_csv(OUTPUT_TEST, index=False)

print(f"\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)
print(f"  Train: {OUTPUT_TRAIN}")
print(f"  Test:  {OUTPUT_TEST}")
print(f"\nRandom state used: {RANDOM_STATE}")
print("\nUse these files for all your models (RF, Logistic Regression, etc.)")