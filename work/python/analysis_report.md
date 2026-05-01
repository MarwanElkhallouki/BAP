# Drift/XAI Analysis Report (Current Run Artifacts)

## Scope
This report analyzes current `output/*_results.npz` artifacts, checks category-level behavior, and investigates whether `metal_nut` differs abnormally.

## Key finding
`metal_nut` itself is not malformed in the result artifacts; a major discrepancy came from notebook filename parsing.
In `results.ipynb`, the previous regex parser could split `metal_nut_*` incorrectly (category contains underscore), which makes `metal_nut` appear inconsistent in plots/tables.
This parser is now fixed to use config-driven category/corruption matching.

## Data coverage
- Parsed rows: 145
- Categories detected: ['bottle', 'carpet', 'leather', 'metal_nut', 'transistor']
- Corruptions detected: ['brightness', 'defocus_blur', 'gaussian_noise', 'held_out_defects', 'jpeg_compression', 'rotation', 'translation']

## Category AUROC summary (all severities/corruptions combined)

```
                mean       min       max
category                                
carpet      0.541581  0.238544  0.764151
transistor  0.728197  0.536667  0.800000
leather     0.760940  0.431134  0.949653
metal_nut   0.850569  0.484190  0.975543
bottle      0.948871  0.778571  1.000000
```

## metal_nut AUROC by corruption

```
                      mean       min       max
corruption                                    
defocus_blur      0.750659  0.484190  0.966403
jpeg_compression  0.817852  0.487154  0.966403
gaussian_noise    0.827404  0.656126  0.966403
brightness        0.912220  0.785573  0.966403
held_out_defects  0.957016  0.957016  0.957016
rotation          0.959363  0.952322  0.966403
translation       0.970973  0.966403  0.975543
```

## Mean AUROC by category x corruption

```
corruption  brightness  defocus_blur  gaussian_noise  held_out_defects  jpeg_compression  rotation  translation
category                                                                                                       
bottle          0.9131        0.9536          0.9409            0.9929            0.9567    0.9875       0.9818
carpet          0.5612        0.4330          0.5237            0.5677            0.6333    0.5378       0.5777
leather         0.6923        0.8724          0.7648            0.8371            0.6762    0.7799       0.8181
metal_nut       0.9122        0.7507          0.8274            0.9570            0.8179    0.9594       0.9710
transistor      0.6708        0.7675          0.7412            0.7404            0.7288    0.7325       0.7311
```

## Interpretation
1. `metal_nut` is actually among higher-performing categories in current artifacts, not failing globally.
2. Lower `metal_nut` scores are concentrated in blur/compression-heavy corruptions, which is plausible for fine-texture parts.
3. The major "different output" symptom is explained by filename parsing mismatch, now corrected.

## Additional sanity checks recommended
1. Re-open `results.ipynb` and re-run all cells after parser fix.
2. Compare clean-test AUROC per category from checkpoints to verify ranking consistency with stream outputs.
3. If needed, retune `metal_nut` train/holdout defect split (currently train=[bent,color], holdout=[flip,scratch]) only if domain intent requires stricter novelty.
