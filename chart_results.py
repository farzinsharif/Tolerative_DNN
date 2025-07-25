import os
import pandas as pd
from openpyxl import Workbook
from collections import defaultdict

# Define the directory containing the results
results_dir = './results'

# Define the BER values and their order
ber_order = ['1x10^-5', '5x10^-5', '1x10^-4', '5x10^-4']
ber_5e6_row = {
    'BER_power': '5x10^-6',
    'Accuracy': 0.7551,
    'Tacc': 0.947,
    'Precision': 0.756340938042665,
    'Recall': 0.7551,
    'Confidence': 0.866746604442596,
    'Sub-Confidence': 0.783372938632965,
    'Acc_50': 0.738900005817413,
    'MAX': 0.7445,
    'original': 0.7445,
    'zop': 0.7445
}

# Define the additional columns data
additional_columns = {
    '1x10^-5': {'MAX': 0.7545, 'original': 0.7185, 'zop': 0.7334},
    '5x10^-5': {'MAX': 0.7503, 'original': 0.52466, 'zop': 0.6692},
    '1x10^-4': {'MAX': 0.743, 'original': 0.2891, 'zop': 0.5069},
    '5x10^-4': {'MAX': 0.6169, 'original': 0.1, 'zop': 0.1}
}

def process_csv_file(file_path):
    """Process a single CSV file and return aggregated data."""
    try:
        df = pd.read_csv(file_path)
        
        # Group by BER_power and calculate mean for all columns except Iteration
        if 'Iteration' in df.columns:
            aggregated = df.drop(columns=['Iteration']).groupby('BER_power').mean().reset_index()
        else:
            aggregated = df.groupby('BER_power').mean().reset_index()
        
        return aggregated
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_sheet_info(filename):
    """Extract pruning and hrank percentages from filename and return as tuple (prune_pct, hrank_pct, sheet_name)"""
    parts = filename.split('_')
    if len(parts) >= 4:
        # Extract prune percentage (remove % and convert to int)
        prune_part = parts[1].replace('%_Prune', '').replace('%prune', '')
        prune_pct = int(prune_part.replace('%', ''))
        
        # Extract hrank percentage (remove % and convert to int)
        hrank_part = parts[3].replace('%_Hrank.csv', '').replace('%hrank.csv', '')
        hrank_pct = int(hrank_part.replace('%', ''))
        
        sheet_name = f"{prune_pct}p_{hrank_pct}h"
        return (prune_pct, hrank_pct, sheet_name)
    return None

def main():
    # Dictionary to store results with sorting information
    results_info = []
    
    # Walk through the results directory
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            # Only process files with the specific pattern: results_X%_Prune_Y%_Hrank.csv
            if (file.startswith('results_') and 
                file.endswith('_Hrank.csv') and 
                ('%_Prune_' in file or '%prune_' in file)):
                
                # Get the sheet info
                sheet_info = get_sheet_info(file)
                if not sheet_info:
                    print(f"Skipping file with unexpected name format: {file}")
                    continue
                
                # Process the CSV file
                file_path = os.path.join(root, file)
                aggregated_data = process_csv_file(file_path)
                
                if aggregated_data is not None:
                    prune_pct, hrank_pct, sheet_name = sheet_info
                    results_info.append({
                        'prune_pct': prune_pct,
                        'hrank_pct': hrank_pct,
                        'sheet_name': sheet_name,
                        'data': aggregated_data
                    })
    
    # Sort the results first by prune percentage, then by hrank percentage
    results_info.sort(key=lambda x: (x['prune_pct'], x['hrank_pct']))
    
    # Create a new Excel workbook
    with pd.ExcelWriter('aggregated_result.xlsx', engine='openpyxl') as writer:
        for result in results_info:
            final_df = result['data']
            
            # Convert BER_power to string for consistent ordering
            final_df['BER_power'] = final_df['BER_power'].astype(str)
            
            # Add additional columns
            final_df['MAX'] = final_df['BER_power'].map(lambda x: additional_columns.get(x, {}).get('MAX', ''))
            final_df['original'] = final_df['BER_power'].map(lambda x: additional_columns.get(x, {}).get('original', ''))
            final_df['zop'] = final_df['BER_power'].map(lambda x: additional_columns.get(x, {}).get('zop', ''))
            
            # Reorder rows according to ber_order
            final_df['BER_power'] = pd.Categorical(final_df['BER_power'], categories=ber_order, ordered=True)
            final_df = final_df.sort_values('BER_power')
            
            # Add the 5x10^-6 row at the beginning
            ber_5e6_df = pd.DataFrame([ber_5e6_row])
            final_df = pd.concat([ber_5e6_df, final_df], ignore_index=True)
            
            # Write to Excel sheet
            # Truncate sheet name to 31 characters (Excel limit)
            final_df.to_excel(writer, sheet_name=result['sheet_name'][:31], index=False)
    
    print("Aggregation complete. Results saved to 'aggregated_result.xlsx'")

if __name__ == '__main__':
    main()