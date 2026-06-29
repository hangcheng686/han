# Chewinggum Localization Improvement Design

Date: 2026-06-29

## Goal

Improve pixel-level anomaly localization on the VisA `chewinggum` category in this AdaCLIP codebase.

The selected strategy is:

1. Pretrain on `MVTec + VisA(without chewinggum)` to learn general anomaly localization.
2. Finetune on `chewinggum` in two stages:
   - Stage 1: normal-only adaptation to the chewinggum appearance domain.
   - Stage 2: short supervised finetuning with chewinggum anomaly images and masks.

This intentionally prioritizes practical localization quality over strict adherence to the original zero-shot / normal-only setting.

## Current Problem

The current `visa_chewinggum_split` setup performs poorly for localization because:

1. Official VisA `train` for `chewinggum` is normal-only, so pixel-level anomaly supervision is absent.
2. The trainer optimizes both image-level and pixel-level objectives, but the current chewinggum-only run does not give the segmentation branch enough anomaly signal.
3. Small local defects are further weakened by whole-image resizing and the current inference-time smoothing.

The expected symptom is exactly what was observed: image-level anomaly detection is partially usable, while pixel-level heatmaps remain diffuse and fail to highlight the defect region.

## Scope

This design covers:

1. Dataset aliases and split logic needed for multi-stage training.
2. Training flow changes for pretraining and two-stage chewinggum finetuning.
3. Validation and model-selection changes to favor localization metrics.
4. Inference changes needed to avoid over-smoothing small defects.

This design does not cover:

1. Large-scale architecture replacement.
2. New model families outside AdaCLIP.
3. Multi-object chewinggum segmentation or instance-level labeling.

## Data Plan

### Phase A: General Localization Pretraining

Training set:

1. `mvtec`
2. `visa_wo_chewinggum`

Purpose:

1. Learn anomaly-region localization from diverse industrial defects with masks.
2. Preserve the current AdaCLIP training interface as much as possible.

### Phase B: Chewinggum Stage 1 Adaptation

Training set:

1. official chewinggum `train` normal images only

Validation set:

1. a dedicated chewinggum validation split containing both normal and anomaly images

Purpose:

1. Adapt the prompt and anomaly representation to chewinggum-specific normal appearance.
2. Reduce domain gap before supervised anomaly-mask finetuning.

### Phase C: Chewinggum Stage 2 Localization Finetuning

Training set:

1. chewinggum anomaly images with masks
2. chewinggum normal images

Validation set:

1. chewinggum validation split kept separate from training

Test set:

1. chewinggum holdout test split

Purpose:

1. Explicitly teach the localization branch where anomaly regions are.
2. Select the best checkpoint using pixel-level metrics instead of only image-level behavior.

## Chewinggum Split Design

Create a dedicated split logic in `dataset/visa.py` for localization-focused chewinggum runs.

The split should be deterministic and reusable through named dataset aliases.

Required aliases:

1. `visa_wo_chewinggum`
2. `chewinggum_stage1_train`
3. `chewinggum_stage1_val`
4. `chewinggum_stage2_train`
5. `chewinggum_stage2_val`
6. `chewinggum_test`

### Split Rules

1. `visa_wo_chewinggum`
   - includes all VisA categories except `chewinggum`
   - uses the existing official VisA split behavior for those categories

2. `chewinggum_stage1_train`
   - uses official chewinggum normal training images only

3. `chewinggum_stage1_val`
   - uses a fixed validation subset from chewinggum test data
   - must contain both normal and anomaly samples

4. `chewinggum_stage2_train`
   - uses the remaining chewinggum normal data plus a supervised subset of anomaly samples and their masks
   - excludes all validation and test samples

5. `chewinggum_stage2_val`
   - fixed validation set with normal + anomaly + masks

6. `chewinggum_test`
   - fixed holdout test set with normal + anomaly + masks

### Practical Split Policy

Use the official VisA chewinggum test pool and partition it deterministically into validation and test subsets.

Recommended policy:

1. Keep official normal `train` unchanged for stage 1.
2. Split official anomaly `test` samples into:
   - stage 2 train subset
   - validation subset
   - test subset
3. Split official normal `test` samples into:
   - validation subset
   - test subset

Implementation should use a fixed random seed or fixed sorted slicing so repeated runs produce the same split.

## Training Flow

### Phase A: Pretraining

Entry:

1. `train.py`

Config:

1. `training_data = ["mvtec", "visa_wo_chewinggum"]`
2. validation data should be one of the localization-capable datasets already supported by the repo

Purpose:

1. Produce a general anomaly-localization checkpoint for later chewinggum adaptation.

### Phase B: Stage 1 Finetuning

Load:

1. Phase A checkpoint

Config:

1. `training_data = chewinggum_stage1_train`
2. `testing_data = chewinggum_stage1_val`

Purpose:

1. Learn chewinggum-specific normal appearance without immediately overfitting anomaly masks.

### Phase C: Stage 2 Finetuning

Load:

1. best checkpoint from Phase B

Config:

1. `training_data = chewinggum_stage2_train`
2. `testing_data = chewinggum_stage2_val`

Purpose:

1. Improve actual defect localization using supervised anomaly masks.

Final Evaluation:

1. test the best Phase C checkpoint on `chewinggum_test`

## Hyperparameter Policy

Use a moderate-change default schedule tuned for stability rather than maximum experimentation.

### Recommended Defaults

Phase A:

1. epochs: 20 to 40
2. learning rate: `1e-3`

Phase B:

1. epochs: 5 to 10
2. learning rate: `5e-4`

Phase C:

1. epochs: 10 to 20
2. learning rate: `1e-4` to `5e-4`

Shared:

1. keep `batch_size = 1` unless the existing limitation is fixed
2. keep current AdaCLIP backbone and prompt settings initially
3. do not change architecture in this implementation round

## Validation and Checkpoint Selection

Current training saves the best checkpoint using `f1_px`, which is directionally correct for localization.

For this design:

1. retain pixel-first checkpoint selection
2. log both image-level and pixel-level metrics every validation run
3. treat `P-F1` and `P-AP` as the primary success indicators for chewinggum
4. treat `I-*` metrics as secondary diagnostics

## Inference Change

Current single-image inference uses:

1. `gaussian_filter(..., sigma=4)`

For small defects, this is too aggressive.

The implementation should:

1. expose smoothing sigma as a CLI argument
2. default to a smaller value for chewinggum-focused inference, such as `1`
3. allow `0` to disable smoothing for debugging

## Code Changes

### `dataset/__init__.py`

Add dataset aliases:

1. `visa_wo_chewinggum`
2. `chewinggum_stage1_train`
3. `chewinggum_stage1_val`
4. `chewinggum_stage2_train`
5. `chewinggum_stage2_val`
6. `chewinggum_test`

### `dataset/visa.py`

Add or extend dataset classes to support:

1. VisA without chewinggum
2. deterministic chewinggum stage splits
3. supervised anomaly-mask training subsets

This logic should remain explicit and easy to audit because split leakage would invalidate the results.

### `train.py`

Keep the main training entry if possible, but add enough configuration support to:

1. load a previous checkpoint as initialization for finetuning
2. run the three phases through consistent command-line flags

If this becomes too awkward, add a dedicated orchestration script rather than overloading the current interface with fragile conditionals.

### `test.py`

Add CLI control for:

1. heatmap smoothing sigma

Retain both dataset and single-image evaluation modes.

### Optional New Script

Add a small orchestration script if needed, for example:

1. `scripts/train_chewinggum_localization.py`

Its responsibility would be:

1. run the three phases in order
2. manage checkpoint handoff between phases
3. keep the base `train.py` simpler

## Risks

1. If the stage 2 anomaly subset is too small, the model may still fail to generalize.
2. If validation and test slicing are not deterministic, results will be noisy and hard to compare.
3. If stage 2 uses too many anomaly samples from the small chewinggum pool, the final test can become too easy or too small.
4. If smoothing remains too strong at inference, visual results may still look worse than the underlying anomaly map quality.

## Success Criteria

The implementation is successful when:

1. all three phases can run end-to-end with explicit commands
2. chewinggum validation and test splits are leakage-free and deterministic
3. final single-image heatmaps visibly respond to actual chewinggum defect regions
4. `P-F1` and/or `P-AP` improve meaningfully relative to the current chewinggum-only baseline

## Verification Plan

Before considering the implementation complete:

1. add tests for each new dataset alias
2. verify stage splits are mutually disjoint
3. verify anomaly-mask subsets contain masks and normal subsets do not fabricate anomaly labels
4. run at least one smoke test for each training phase configuration
5. run single-image inference with multiple smoothing values and compare outputs

## Recommended Implementation Order

1. add dataset aliases and deterministic chewinggum split logic
2. add tests that prove split membership and non-overlap
3. add finetuning/checkpoint-loading support for staged training
4. add optional orchestration for the three-phase workflow
5. add smoothing control to single-image inference
6. run a smoke test for each phase

