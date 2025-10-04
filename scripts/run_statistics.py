#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point for Statistical Analysis
Runs complete analysis pipeline: metrics calculation and report generation
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Execute command and display progress"""
    print(f"\n{'='*80}")
    print(f"  {description}")
    print('='*80)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"  Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Failed: {e}")
        return False


def main():
    print("="*80)
    print("Statistical Analysis for Wearable Device Performance")
    print("="*80)
    
    # Get paths - script is now in scripts/ folder
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    calc_script = script_dir / 'calculate_metrics_statistics.py'
    excel_script = script_dir / 'generate_excel_report.py'
    
    if not calc_script.exists():
        print(f"\nError: Script not found - {calc_script}")
        sys.exit(1)
    
    # Step 1: Calculate metrics
    success = run_command(
        f'"{sys.executable}" "{calc_script}"',
        "Step 1/2: Calculate Metrics"
    )
    
    if not success:
        print("\nError: Metrics calculation failed")
        sys.exit(1)
    
    # Step 2: Generate Excel report
    if excel_script.exists():
        success = run_command(
            f'"{sys.executable}" "{excel_script}"',
            "Step 2/2: Generate Excel Report"
        )
        
        if not success:
            print("\nWarning: Excel generation failed")
            print("Install dependency: pip install openpyxl")
    else:
        print(f"\nWarning: Script not found - {excel_script}")
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    
    results_dir = base_dir / 'statistics_results'
    print(f"\nOutput: {results_dir}")
    print("\nKey files:")
    print("  table1_[channel]_weighted.csv - Performance by channel (4 files)")
    print("  table2_[scenario]_[channel].csv - Performance by scenario group (12 files)")
    print("  table3_channel_comparison.csv - Channel comparison")
    print("  table4_best_models.csv - Best model recommendations")
    print("  statistical_report.xlsx - Excel workbook")
    print("  TABLE_DESCRIPTIONS.txt - Table documentation")
    print("="*80)
    
    # Open results directory
    if sys.platform == 'win32':
        try:
            subprocess.run(['explorer', str(results_dir)], check=False)
        except:
            pass
    elif sys.platform == 'darwin':
        try:
            subprocess.run(['open', str(results_dir)], check=False)
        except:
            pass
    elif sys.platform.startswith('linux'):
        try:
            subprocess.run(['xdg-open', str(results_dir)], check=False)
        except:
            pass


if __name__ == '__main__':
    main()

