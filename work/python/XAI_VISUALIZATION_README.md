# XAI Visualization Notebook Guide

## Overview

The `xai_visualization.ipynb` notebook provides interactive visualization and analysis of Explainable AI (XAI) results from drift detection alarms.

It helps you understand:
- **What** the model is attending to (Grad-CAM heatmaps)
- **How** pixel importance changes between pre/post-drift (LIME superpixels)
- **Why** alarms are classified as "small" or "large" scale (component scores)

---

## Quick Start

### 1. Prerequisites
Ensure your pipeline was run with checkpoints enabled:

```python
# In config.py
SAVE_XAI_CHECKPOINTS = True
```

Then run the pipeline:
```bash
python run_experiment.py --category carpet --corruption gaussian_noise
```

This generates checkpoint files in `output/xai_checkpoints/`:
- `*.npz` — Binary checkpoint data (images, model state)
- `*.json` — XAI analysis results (Grad-CAM metrics, LIME overlap)

### 2. Open the Notebook
```bash
jupyter notebook xai_visualization.ipynb
```

### 3. Run Sections in Order
1. **Setup** — Loads utilities and discovers checkpoints
2. **Checkpoint Deep Dive** — Visualize individual alarms
3. **Aggregate Dashboard** — System-wide statistics
4. **Exploration Interface** — Manual navigation

---

## Sections Explained

### Section 1: Setup
- Configures Jupyter environment
- Discovers all checkpoints in `output/xai_checkpoints/`
- Defines helper functions for loading and rendering

**Output:** List of available checkpoints

```
Found 12 checkpoint pairs
  - carpet_brightness_ADWIN_001
  - carpet_brightness_DDM_000
  - carpet_gaussian_noise_ADWIN_000
  ...
```

---

### Section 2: Checkpoint Deep Dive

#### 2.1 Grad-CAM Visualization
Renders **pre/post-drift heatmaps overlaid on images**.

**What to look for:**
- **Red regions** = areas the model focuses on for defect detection
- **Change intensity** = attribution shift (indicates drift strength)
- **Localization accuracy** = does it focus on actual anomalies?

**Example interpretation:**
```
Pre-drift:  Model focuses on texture boundaries (correct)
Post-drift: Model focuses on image corners (incorrect - drift!)
Change:     Large shift in attribution → "large" scale drift
```

#### 2.2 LIME Visualization
Renders **superpixel segmentation overlays**.

**What to look for:**
- Different colored regions = superpixels
- Dense segmentation = fine-grained explanations
- Consistency between pre/post = attribution stability

**Note:** LIME importance scores are not persisted in checkpoints (expensive to store), but the segmentation structure shows how LIME carves up the image space.

#### 2.3 Scale Inference Panel
Displays **detailed rationale for scale prediction** (small vs large).

**Example output:**
```
Alarm Details:
  Detector: ADWIN
  Category: carpet
  Corruption: gaussian_noise
  Severity: 3
  Ground Truth Scale: large

Grad-CAM Analysis:
  Pre-drift mean range: [0.001, 0.234]
  Post-drift mean range: [0.051, 0.456]

LIME Analysis:
  Mean overlap coefficient: 0.312
  Samples analyzed: 10
  Perturbations per sample: 100
  Overlap threshold (large scale): 0.50
```

**How scale is inferred:**
1. **Grad-CAM component:** Attribution shift magnitude → score [0, 1]
2. **LIME component:** Overlap below 0.50 → score [0, 1] (low overlap = high score)
3. **Combined:** Weighted average of both components
4. **Decision:** If combined ≥ 0.50 → "large", else → "small"

---

### Section 3: Aggregate Dashboard

#### Alarm Metrics DataFrame
Loads all checkpoint JSON files and creates a pandas DataFrame.

**Columns:**
- `checkpoint` — Checkpoint name
- `detector` — Drift detector (ADWIN, DDM, etc.)
- `category` — Product category (carpet, bottle, etc.)
- `corruption` — Corruption type (gaussian_noise, rotation, etc.)
- `severity` — Severity level (0-5)
- `drift_type` — Type of drift (corruption, defect, geometric)
- `true_scale` — Ground truth scale from metadata (small/large)
- `lime_overlap` — Mean LIME overlap coefficient [0, 1]

#### LIME Overlap Distribution
**Histogram showing:**
- X-axis: LIME overlap coefficient (0 = no overlap, 1 = perfect overlap)
- Red dashed line: Decision threshold at 0.50
- Overlaps < 0.50 indicate substantial attribution shift → inferred "large" scale

#### Alarm Summary Statistics
**Tables grouped by:**
1. **Detector** — Performance per detector
2. **Category** — Patterns per product type
3. **True Scale** — Behavior for ground-truth small vs large

**Key metrics:**
- `count` — Number of alarms
- `mean`, `std`, `min`, `max` — LIME overlap statistics

---

### Section 4: Exploration Interface

**Manual checkpoint selection:**

```python
checkpoint_idx = 0  # Change to select different checkpoint
```

Then re-run visualization cells to view new alarm.

**Quick reference table** lists all checkpoint names with indices for easy lookup.

---

## Common Workflows

### Workflow 1: Inspect One Failing Detector
1. Go to Aggregate Dashboard
2. Find detector with low "correctness"
3. Note checkpoint names for that detector
4. In Exploration section, set `checkpoint_idx` to view them
5. Examine Grad-CAM change maps and LIME overlap

### Workflow 2: Understand False Positives
1. Aggregate Dashboard → filter `true_scale = None` (false positives)
2. Check if LIME overlap is high (> 0.50)
3. Inspect Grad-CAM: is attribution actually changing?
4. Possible causes: threshold too sensitive, detector too aggressive

### Workflow 3: Analyze Category-Specific Issues
1. Aggregate Dashboard → group by category
2. Find category with lowest mean LIME overlap
3. Hypothesize: noisy data, ambiguous corruption, etc.
4. Deep dive into 2-3 checkpoints from that category

### Workflow 4: Scale Inference Validation
1. Create comparison: true scale vs inferred scale
2. Check agreement rate (%) for each detector
3. For misclassifications:
   - Low combined score (0.3-0.7) = borderline cases
   - High score (> 0.9) but wrong = possible threshold issue
   - Component disagreement (gradcam high, lime low) = mixed signals

---

## Data Definitions

### LIME Overlap Coefficient
**Formula:** `|top_k_pre ∩ top_k_post| / min(|top_k_pre|, |top_k_post|)`

- **Range:** [0, 1]
- **Interpretation:**
  - 0.9–1.0 = Stable attribution (small drift)
  - 0.5–0.9 = Moderate attribution shift
  - 0.0–0.5 = Major attribution change (large drift)

### Grad-CAM Change Map
**Formula:** `post_mean_heatmap - pre_mean_heatmap`

- **Positive values** = More activation post-drift
- **Negative values** = Less activation post-drift
- **Magnitude** = Intensity of shift

### Scale Inference Scores

**Grad-CAM component:**
```
score = min(1.0, mean_abs_change / THRESHOLD)
default THRESHOLD = 0.10
```

**LIME component:**
```
score = max(0.0, (THRESHOLD - overlap) / THRESHOLD)
default THRESHOLD = 0.50
```

**Combined:**
```
combined = (gradcam_score * gradcam_weight + lime_score * lime_weight) / (gradcam_weight + lime_weight)
default weights = 1.0 each
```

---

## Troubleshooting

### Issue: "No checkpoints found"
**Solution:** Ensure `SAVE_XAI_CHECKPOINTS = True` in `config.py` and pipeline completed successfully.

### Issue: Grad-CAM heatmaps appear empty/uniform
**Solution:** Check if corruption actually triggered drift. Severe corruption → large heatmap changes.

### Issue: LIME overlap always ~0.5
**Solution:** Normal! Threshold is set at 0.50 for balanced classification. Use standard deviation to assess variability.

### Issue: Notebook runs slowly
**Solution:** Reduce number of samples visualized in deep dive sections (e.g., `n_samples=2` instead of 3).

---

## Customization

### Change LIME Perturbation Count
```python
# In section 2.3, before running XAI:
LIME_N_PERTURBATIONS = 256  # default 100
```

### Filter Alarms by Detector
```python
detector_name = 'ADWIN'
df_filtered = df_alarms[df_alarms['detector'] == detector_name]
```

### Export Visualizations
```python
# After rendering a plot:
plt.savefig(f'gradcam_{checkpoint_idx}.png', dpi=300, bbox_inches='tight')
```

---

## References

- **Grad-CAM:** Selvaraju et al., *Grad-CAM: Why did you say that?* (ICCV 2017)
- **LIME:** Ribeiro et al., *"Why should I trust you?" Explaining the predictions of any classifier* (KDD 2016)
- **Drift Detection:** Gomes et al., *Adaptive learning in non-stationary and imbalanced data streams* (Neurocomputing 2017)

---

## Next Steps

1. **Run multiple experiments** with different corruptions/categories
2. **Compare detectors** using aggregate dashboard
3. **Tune thresholds** based on scale inference accuracy
4. **Export visualizations** for reports/presentations
5. **Investigate misclassifications** systematically

Good luck with your XAI analysis! 📊
