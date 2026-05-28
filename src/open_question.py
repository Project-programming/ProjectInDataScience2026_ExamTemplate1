import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt



import joblib
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay



#merge patient data: gender and age with the features.csv for extended baseline 
 

train_df = pd.read_csv("features_extended_train.csv")
val_df   = pd.read_csv("features_extended_validation.csv")
test_df  = pd.read_csv("features_extended_testing.csv")

#merge all three features csv extended files 
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total images after combining splits: {len(all_df)}")

metadata = pd.read_csv("/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/metadata-merged(in).csv")

#columns exist
print(metadata[['patient_id', 'age', 'gender']].head())
#actually combining the patient id and age and gender columns with the features_extended

all_df['patient_id'] = all_df['img_id'].str.extract(r'PAT_(\d+)_').astype(int)


#  patient number for the main dataframe (makes it an int64)
all_df['patient_id'] = all_df['img_id'].str.extract(r'PAT_(\d+)_').astype(int)


metadata['patient_id'] = metadata['patient_id'].astype(str).str.replace('PAT_', '').astype(int)


merged_df = all_df.merge(metadata[['patient_id', 'age', 'gender']],
                          on='patient_id', how='inner')

# results
print(f"Rows after metadata merge: {len(merged_df)}")
print(f"Gender value counts:\n{merged_df['gender'].value_counts()}")
print(f"Age stats:\n{merged_df['age'].describe()}")



# the saved model and scaler, generate predictions 
scaler = joblib.load("/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/extended_baseline_scaler.pkl")
model  = joblib.load("/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/extended_baseline_random_forest.pkl")

feature_cols = [
    'asymmetry_score', 'border_irregularity',
    'frac_white', 'frac_red', 'frac_light_brown',
    'frac_dark_brown', 'frac_blue_gray', 'frac_black',
    'n_distinct_colors', 'color_entropy', 'off_palette_dist',
    'diameter_px', 'hair_coverage'
]


# Remove any features not in the CSV 
feature_cols = [c for c in feature_cols if c in merged_df.columns]

X      = scaler.transform(merged_df[feature_cols].values)
y_true = merged_df['is_cancer'].values
y_prob = model.predict_proba(X)[:, 1]   # cancer probability for each image

merged_df['y_prob'] = y_prob
merged_df['y_pred'] = model.predict(X)

# ── Step 4: Helper function for subgroup metrics ──────────────────────────────
def subgroup_metrics(df, group_name):
    """Compute AUC and key metrics for a subgroup dataframe."""
    if len(df) < 10:
        print(f"  {group_name}: too few samples ({len(df)}) — skipping")
        return None

    y_t = df['is_cancer'].values
    y_p = df['y_prob'].values
    y_pred = df['y_pred'].values

    # Need both classes present to compute AUC
    if len(np.unique(y_t)) < 2:
        print(f"  {group_name}: only one class present — cannot compute AUC")
        return None

    auc  = roc_auc_score(y_t, y_p)
    report = classification_report(y_t, y_pred,
                                   target_names=['Benign', 'Cancer'],
                                   output_dict=True)

    print(f"\n── {group_name} (n={len(df)}, cancer={y_t.sum()}) ──")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Recall:    {report['Cancer']['recall']:.4f}")
    print(f"  Precision: {report['Cancer']['precision']:.4f}")
    print(f"  F1:        {report['Cancer']['f1-score']:.4f}")
    print(f"  Accuracy:  {report['accuracy']:.4f}")

    return {
        'group':     group_name,
        'n':         len(df),
        'n_cancer':  int(y_t.sum()),
        'auc':       round(auc, 4),
        'recall':    round(report['Cancer']['recall'], 4),
        'precision': round(report['Cancer']['precision'], 4),
        'f1':        round(report['Cancer']['f1-score'], 4),
        'accuracy':  round(report['accuracy'], 4),
    }

#  Bootstrap confidence interval for AUC 
def bootstrap_auc(df, n_bootstrap=1000, seed=42):
    """Return 95% bootstrap CI for AUC."""
    rng   = np.random.default_rng(seed)
    y_t   = df['is_cancer'].values
    y_p   = df['y_prob'].values
    aucs  = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(df), size=len(df))
        if len(np.unique(y_t[idx])) < 2:
            continue   # skip resamples with only one class
        aucs.append(roc_auc_score(y_t[idx], y_p[idx]))




    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return round(lower, 4), round(upper, 4)

# gender
print("\n══════════════════════════════════")
print("GENDER ANALYSIS")
print("══════════════════════════════════")

# Check what gender values look like in your metadata
# Common formats: 'MALE'/'FEMALE', 'M'/'F', 1/0
print("Unique gender values:", merged_df['gender'].unique())

male_df   = merged_df[merged_df['gender'].str.upper().isin(['MALE', 'M'])]
female_df = merged_df[merged_df['gender'].str.upper().isin(['FEMALE', 'F'])]

male_metrics   = subgroup_metrics(male_df,   "Male")
female_metrics = subgroup_metrics(female_df, "Female")

if male_metrics and female_metrics:
    male_ci   = bootstrap_auc(male_df)
    female_ci = bootstrap_auc(female_df)
    print(f"\nMale   AUC 95% CI: ({male_ci[0]}, {male_ci[1]})")
    print(f"Female AUC 95% CI: ({female_ci[0]}, {female_ci[1]})")

# age
print("\n══════════════════════════════════")
print("AGE ANALYSIS")
print("══════════════════════════════════")

print(f"Age distribution:\n{merged_df['age'].describe()}")

young_df = merged_df[merged_df['age'] <  50]
older_df = merged_df[merged_df['age'] >= 50]

young_metrics = subgroup_metrics(young_df, "Under 50")
older_metrics = subgroup_metrics(older_df, "50 and over")

if young_metrics and older_metrics:
    young_ci = bootstrap_auc(young_df)
    older_ci = bootstrap_auc(older_df)
    print(f"\nUnder 50  AUC 95% CI: ({young_ci[0]}, {young_ci[1]})")
    print(f"50+       AUC 95% CI: ({older_ci[0]}, {older_ci[1]})")

# ── Step 8: Summary table ─────────────────────────────────────────────────────
all_metrics = [m for m in [male_metrics, female_metrics,
                             young_metrics, older_metrics] if m]
summary_df = pd.DataFrame(all_metrics)
print("\n══════════════════════════════════")
print("SUMMARY TABLE")
print("══════════════════════════════════")
print(summary_df.to_string(index=False))
summary_df.to_csv("open_question_results.csv", index=False)

#   AUC comparison bar chart  
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gender plot
gender_groups = ['Male', 'Female']
gender_aucs   = [male_metrics['auc'], female_metrics['auc']] if (male_metrics and female_metrics) else []
gender_cis    = [male_ci, female_ci] if (male_metrics and female_metrics) else []

if gender_aucs:
    bars = axes[0].bar(gender_groups, gender_aucs, color=['steelblue', 'coral'],
                       width=0.4, zorder=3)
    # Add error bars from bootstrap CI
    yerr_low  = [gender_aucs[i] - gender_cis[i][0] for i in range(2)]
    yerr_high = [gender_cis[i][1] - gender_aucs[i] for i in range(2)]
    axes[0].errorbar(gender_groups, gender_aucs,
                     yerr=[yerr_low, yerr_high],
                     fmt='none', color='black', capsize=6, linewidth=2)
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random (AUC=0.5)')
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_title('Model AUC by Gender')
    axes[0].legend()
    axes[0].bar_label(bars, fmt='%.3f', padding=4)
    axes[0].grid(axis='y', alpha=0.3, zorder=0)






