# Scripts Directory

This directory contains scripts for model testing, prediction generation, and statistical analysis.

---

## Quick Start

### Generate Predictions for All Models
```bash
python scripts/test_all_models_predictions.py
```

### Run Statistical Analysis
```bash
python scripts/run_statistics.py
```

---

## Scripts Overview

### Model Testing and Prediction Generation

#### `test_all_models_predictions.py` - Batch Model Testing

**Purpose**: Batch test all trained models and generate detailed predictions with metadata for paper reproduction and analysis.

**Usage**:
```bash
# Test all models in models/ directory
python scripts/test_all_models_predictions.py

# Test specific models
python scripts/test_all_models_predictions.py --models resnet-ring1-hr-all-ir mamba2-ring2-spo2-all-irred
```

**Input**: 
- Trained models in `models/` directory
- Model configurations (from model folder or config/train/)
- Raw data from specified data path

**Output**: 
- Predictions with complete metadata saved to `predictions/<exp_name>/<fold>.csv`

**Output Format**:
```csv
prediction,target,subject_id,scenario,start_time,end_time,task,exp_name
76.68,101.73,00023,sitting,1742822979.47,1742823009.47,hr,resnet-ring1-hr-all-ir
```

**Features**:
- Automatically discovers all trained models
- Tests across all 5 folds
- Collects metadata: subject_id, scenario, timestamps
- Supports all tasks: HR, RR, SpO2, BP (SBP and DBP)
- Handles model-specific requirements (e.g., batch size for Mamba)
- Compatible with all model types (ResNet, InceptionTime, Transformer, Mamba)

**Key Implementation**:
- `DetailedDataset` class: Custom dataset with metadata collection
- `custom_collate()`: Handle metadata in DataLoader batches
- Automatic config compatibility handling (removes incompatible parameters)

---

### Statistical Analysis

### 1. `run_statistics.py` - Main Entry Point

**Purpose**: Orchestrates the complete analysis workflow

**Usage**:
```bash
python scripts/run_statistics.py
```

**Process**:
1. Calls `calculate_metrics_statistics.py` to compute metrics
2. Calls `generate_excel_report.py` to create Excel workbook
3. Opens results directory upon completion

---

### 2. `calculate_metrics_statistics.py` - Metrics Calculation

**Purpose**: Computes performance metrics from prediction CSV files

**Input**: `predictions/` directory containing fold-wise prediction results

**Output**: Statistical tables in `statistics_results/` directory

**Metrics Computed**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- Pearson Correlation Coefficient

**All metrics reported as**: mean ± standard deviation (across 5 folds)

**Generated Tables**:

- **Table 1** (4 files): `table1_[channel]_weighted.csv`
  - Performance for each channel across all scenarios
  - Weighted average by sample size
  - Channels: ir, irimu, irred, irredimu

- **Table 2** (12 files): `table2_[scenario_group]_[channel].csv`
  - Performance for each scenario group and channel combination
  - Scenario groups: motion, spo2, stationary
  - Example: `table2_stationary_irred.csv`

- **Table 3** (1 file): `table3_channel_comparison.csv`
  - Side-by-side comparison of all channels
  - Pivoted format for easy comparison

- **Table 4** (1 file): `table4_best_models.csv`
  - Best model recommendations
  - Based on lowest weighted MAE
  - For each task+channel+scenario_group+ring combination

- **Raw Data** (2 files):
  - `raw_aggregated.csv` - 5-fold averaged data (basis for all tables)
  - `raw_fold_level.csv` - Individual fold results

- **Documentation**: `TABLE_DESCRIPTIONS.txt` - Detailed description of each table

---

### 3. `generate_excel_report.py` - Excel Report Generator

**Purpose**: Consolidates all CSV tables into a single Excel workbook

**Input**: CSV files in `statistics_results/` directory

**Output**: `statistical_report.xlsx` with 20+ worksheets

**Requirements**: `pip install openpyxl`

**Worksheet Structure**:
- README - Overview and table descriptions
- All-IR/IRIMU/IRRED/IRREDIMU - Table 1 sheets
- Motion/Spo2/Stationary-[CHANNEL] - Table 2 sheets (12 total)
- Channel Comparison - Table 3
- Best Models - Table 4
- Raw Aggregated, Raw Fold Level - Raw data

---

## Analysis Dimensions

### Models
- inception-time
- mamba2
- resnet
- transformer

### Tasks
- hr (Heart Rate)
- resp_rr (Respiratory Rate)
- BP_sys (Systolic Blood Pressure)
- BP_dia (Diastolic Blood Pressure)
- spo2 (Blood Oxygen Saturation)

### Rings
- ring1
- ring2

### Channels (Input Configurations)
- ir (Infrared only)
- irimu (Infrared + IMU)
- irred (Infrared + Red)
- irredimu (Infrared + Red + IMU)

### Scenarios and Scenario Groups

Original scenarios are categorized into three groups based on activity characteristics:

#### Scenario Groups

**Motion** (High-intensity movement scenarios)
- `deepsquat` - Deep squat exercises
- `striding` - Walking/striding activities
- Characteristics: Large body movements, signal instability, challenging for prediction

**SPO2** (Specialized measurement scenario)
- `spo2` - Blood oxygen saturation measurement protocol
- Characteristics: Controlled breathing, specific measurement context

**Stationary** (Low-intensity or static scenarios)
- `sitting` - Seated position
- `talking` - Verbal communication while seated/standing
- `shaking_head` - Head movement
- `standing` - Standing position
- Characteristics: Minimal body movement, stable signals, optimal for prediction

#### Rationale for Grouping

This grouping enables analysis of:
1. **Performance variation** across different activity levels
2. **Model robustness** under motion vs. static conditions
3. **Scenario-specific optimization** for different use cases

Typical performance hierarchy: Stationary > SPO2 > Motion (in terms of accuracy)

---

## Calculation Methods

### Weighted Average
All primary tables use weighted averaging by sample size:

```
metric_weighted = Σ(scenario_metric × scenario_samples) / Σ(scenario_samples)
```

This accounts for differences in sample sizes across scenarios and provides more representative performance estimates.

### Standard Deviation
Reported as the mean of standard deviations across aggregated groups.

---

## Important Notes

1. **Channel Independence**: Each channel is analyzed separately. Different channels represent different input configurations and should not be mixed.

2. **Scenario Groups**: Scenarios are grouped by activity type to facilitate analysis of performance under different conditions.

3. **Weighted Averaging**: All metrics are weighted by sample size to account for imbalanced data across scenarios.

4. **Format**: All metrics reported as `mean±std` format (e.g., `6.16±3.84`).

---

## Example Usage

### For paper reporting:

**Best overall performance**:
- Check `table1_ir_weighted.csv` for each channel's best performance
- Example: Resnet achieved MAE of 5.63±2.76 (IR channel, HR task, Ring1)

**Scenario-specific analysis**:
- Check `table2_stationary_irred.csv` for performance in specific contexts
- Example: In stationary scenarios with IRRED channel, Resnet achieved MAE of 5.09±4.58

**Channel selection**:
- Check `table3_channel_comparison.csv` to compare all channels side-by-side
- Identifies optimal channel configuration for each model

**Model recommendation**:
- Check `table4_best_models.csv` for recommended model given specific requirements

---

## File Naming Convention

- `table1_*` - By channel (primary analysis)
- `table2_*` - By scenario group and channel
- `table3_*` - Channel comparison
- `table4_*` - Best model recommendations
- `raw_*` - Raw data files

---

## Dependencies

```bash
pip install pandas numpy scipy openpyxl
```

---

## Troubleshooting

**Issue**: Excel generation fails with permission error  
**Solution**: Close the Excel file if it's open, then re-run

**Issue**: No data found  
**Solution**: Ensure `predictions/` directory contains Fold-*.csv files

**Issue**: Missing openpyxl module  
**Solution**: Install via `pip install openpyxl`

---

Last updated: 2025-10-04
