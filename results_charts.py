import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_accuracy_plots(xls_file_path, output_dir):
    """Generate accuracy comparison plots for each sheet in the Excel file."""
    # Load the Excel file
    xls = pd.ExcelFile(xls_file_path)
    
    # Process each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Extract prune and hrank from sheet name
        prune_percent = sheet_name.split('_')[0].replace('p', '%Prune')
        hrank_percent = sheet_name.split('_')[1].replace('h', '%Hrank')
        title_suffix = f"{prune_percent}_{hrank_percent}"
        
        # Create the plot
        create_single_plot(df, sheet_name, title_suffix, output_dir)

def create_single_plot(df, sheet_name, title_suffix, output_dir):
    """Create and save a single accuracy comparison plot."""
    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # Get BER values and convert to strings for x-axis labels
    ber_values = df['BER_power'].astype(str)
    x = range(len(ber_values))
    width = 0.2  # Width of each bar
    
    # Plot each metric
    plt.bar([i - 1.5*width for i in x], df['MAX'], width, label='MAX')
    plt.bar([i - 0.5*width for i in x], df['Accuracy'], width, label='Proposed Method')
    plt.bar([i + 0.5*width for i in x], df['original'], width, label='Original')
    plt.bar([i + 1.5*width for i in x], df['zop'], width, label='ZOP')
    
    # Customize the plot
    plt.title(f"Bit error rate vs. Accuracy\n{title_suffix}", fontsize=12)
    plt.xlabel('Bit Error Rate (BER)', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(x, ber_values)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"accuracy_bar_plot_{sheet_name}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Main function to coordinate the plotting process."""
    # Define paths
    current_dir = os.getcwd()
    xls_file_path = os.path.join(current_dir, 'results', 'aggregated_result.xlsx')
    output_dir = os.path.join(current_dir, 'results', 'accuracy_plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the plots
    generate_accuracy_plots(xls_file_path, output_dir)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main()