import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score, classification_report

# ── Step 1: Load and combine all feature splits ───────────────────────────────
train_df = pd.read_csv("features_extended_train.csv")
val_df   = pd.read_csv("features_extended_validation.csv")
test_df  = pd.read_csv("features_extended_testing.csv")

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total images after combining splits: {len(all_df)}")

# ── Step 2: Merge with metadata (age + gender) ────────────────────────────────
metadata = pd.read_csv("/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/metadata-merged(in).csv")



# Extract numeric patient ID from img_id like "PAT_441_2868_663" → 441
all_df['patient_id'] = all_df['img_id'].str.extract(r'PAT_(\d+)_').astype(int)

# forces metadata patient_id to int so types match

metadata['patient_id'] = metadata['patient_id'].str.replace('PAT_', '').astype(int)

merged_df = all_df.merge(
    metadata[['patient_id', 'age', 'gender']],
    on='patient_id',
    how='inner'
)

print(f"Rows after metadata merge: {len(merged_df)}")
print(f"Gender value counts:\n{merged_df['gender'].value_counts()}")
print(f"Age stats:\n{merged_df['age'].describe()}")

# ── Step 3: Load model + scaler, generate predictions ────────────────────────
scaler = joblib.load("/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/extended_baseline_scaler.pkl")
model  = joblib.load("/Users/juliak/Desktop/ProjectInDataScience2026_ExamTemplate1/extended_baseline_random_forest.pkl")

feature_cols = [
    'asymmetry_score', 'border_irregularity',
    'frac_white', 'frac_red', 'frac_light_brown',
    'frac_dark_brown', 'frac_blue_gray', 'frac_black',
    'n_distinct_colors', 'color_entropy', 'off_palette_dist',
    'diameter_px', 'hair_coverage'
]
# Keep only columns that actually exist in the CSV
feature_cols = [c for c in feature_cols if c in merged_df.columns]

X      = scaler.transform(merged_df[feature_cols].values)
y_true = merged_df['is_cancer'].values
y_prob = model.predict_proba(X)[:, 1]

merged_df['y_prob'] = y_prob
merged_df['y_pred'] = model.predict(X)

# ── Step 4: Subgroup metrics helper ──────────────────────────────────────────
def subgroup_metrics(df, group_name):
    if len(df) < 10:
        print(f"  {group_name}: too few samples ({len(df)}) — skipping")
        return None
    y_t    = df['is_cancer'].values
    y_p    = df['y_prob'].values
    y_pred = df['y_pred'].values
    if len(np.unique(y_t)) < 2:
        print(f"  {group_name}: only one class present — cannot compute AUC")
        return None

    auc    = roc_auc_score(y_t, y_p)
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

# ── Step 5: Bootstrap CI for AUC ─────────────────────────────────────────────
def bootstrap_auc(df, n_bootstrap=1000, seed=42):
    rng  = np.random.default_rng(seed)
    y_t  = df['is_cancer'].values
    y_p  = df['y_prob'].values
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(df), size=len(df))
        if len(np.unique(y_t[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_t[idx], y_p[idx]))
    return round(np.percentile(aucs, 2.5), 4), round(np.percentile(aucs, 97.5), 4)

# ── Step 6: Gender analysis ───────────────────────────────────────────────────
print("\n══════════════════════════════════")
print("GENDER ANALYSIS")
print("══════════════════════════════════")
print("Unique gender values:", merged_df['gender'].unique())

male_df   = merged_df[merged_df['gender'].str.upper().isin(['MALE',   'M'])]
female_df = merged_df[merged_df['gender'].str.upper().isin(['FEMALE', 'F'])]

male_metrics   = subgroup_metrics(male_df,   "Male")
female_metrics = subgroup_metrics(female_df, "Female")

male_ci = female_ci = None
if male_metrics and female_metrics:
    male_ci   = bootstrap_auc(male_df)
    female_ci = bootstrap_auc(female_df)
    print(f"\nMale   AUC 95% CI: ({male_ci[0]}, {male_ci[1]})")
    print(f"Female AUC 95% CI: ({female_ci[0]}, {female_ci[1]})")

# ── Step 7: Age analysis ──────────────────────────────────────────────────────
print("\n══════════════════════════════════")
print("AGE ANALYSIS")
print("══════════════════════════════════")
print(f"Age distribution:\n{merged_df['age'].describe()}")

young_df = merged_df[merged_df['age'] <  50]
older_df = merged_df[merged_df['age'] >= 50]

young_metrics = subgroup_metrics(young_df, "Under 50")
older_metrics = subgroup_metrics(older_df, "50 and over")

young_ci = older_ci = None
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

# ── Step 9: AUC bar charts (gender + age side by side) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

def plot_auc_bars(ax, groups, aucs, cis, colors, title):
    """Draw a bar chart with bootstrap CI error bars."""
    bars = ax.bar(groups, aucs, color=colors, width=0.4, zorder=3)
    yerr_low  = [aucs[i] - cis[i][0] for i in range(len(aucs))]
    yerr_high = [cis[i][1] - aucs[i] for i in range(len(aucs))]
    ax.errorbar(groups, aucs,
                yerr=[yerr_low, yerr_high],
                fmt='none', color='black', capsize=6, linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random (AUC=0.5)')
    ax.set_ylim(0, 1)
    ax.set_ylabel('AUC-ROC')
    ax.set_title(title)
    ax.legend()
    ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.grid(axis='y', alpha=0.3, zorder=0)

# Gender subplot
if male_metrics and female_metrics and male_ci and female_ci:
    plot_auc_bars(
        axes[0],
        groups=['Male', 'Female'],
        aucs=[male_metrics['auc'], female_metrics['auc']],
        cis=[male_ci, female_ci],
        colors=['steelblue', 'coral'],
        title='Model AUC by Gender'
    )

# Age subplot
if young_metrics and older_metrics and young_ci and older_ci:
    plot_auc_bars(
        axes[1],
        groups=['Under 50', '50 and over'],
        aucs=[young_metrics['auc'], older_metrics['auc']],
        cis=[young_ci, older_ci],
        colors=['mediumseagreen', 'mediumpurple'],
        title='Model AUC by Age Group'
    )

plt.tight_layout()
plt.savefig("open_question_auc_comparison.png", dpi=150)
print("\nChart saved to open_question_auc_comparison.png")