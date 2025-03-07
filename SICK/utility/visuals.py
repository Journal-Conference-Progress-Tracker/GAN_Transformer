
import matplotlib.pyplot as plt
import numpy as np

def save_table_as_image(
        summary_df,
        file_name="accuracy_summary.png",
        title="Accuracy Comparisons: Real Only v.s. After Concatenation",
        bar1_name="Real Only Accuracy",
        bar2_name="After Concatenation Accuracy"
    ):
    """
    Save the accuracy results as an image in ACM format.
    """
    # Prepare summary data for table
    summary_data = []

    # Add generation size column if it exists
    if "Generation Size" in summary_df.columns:
        generation_sizes = summary_df["Generation Size"].unique()
        for gen in generation_sizes:
            df_gen = summary_df[summary_df["Generation Size"] == gen]
            for _, row in df_gen.iterrows():
                summary_data.append([gen, row["Train Samples"], row[bar1_name], row[bar2_name]])
    else:
        for _, row in summary_df.iterrows():
            summary_data.append([None, row["Train Samples"], row[bar1_name], row[bar2_name]])

    # Create table header
    table_header = ["Generation Size", "Train Samples", bar1_name, bar2_name]
    
    # Plot the table using matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    # Create the table and display it
    table = ax.table(cellText=summary_data, colLabels=table_header, loc="center", cellLoc="center", colColours=["#f5f5f5"] * len(table_header))

    # Set fonts
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)  # Scale table for better readability

    # Title of the table
    ax.set_title(title, fontweight='bold', fontsize=16, family="Times New Roman")

    # Save the table to a file (image)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)

    print(f"Table has been saved as '{file_name}'")

def display_accuracy_summary(
        summary_df,
        x_label_title="Train Samples",
        y_label="Accuracy",
        title="Accuracy Comparisons: Real Only v.s. After Concatenation",
        file_name="plot.png",
        bar1_name="Real Only Accuracy",
        bar2_name="After Concatenation Accuracy"
    ):
    # Check if the DataFrame contains a "Generation Size" column
    if "Generation Size" in summary_df.columns:
        generation_sizes = summary_df["Generation Size"].unique()
        for gen in generation_sizes:
            # Filter the DataFrame for the current generation size
            df_gen = summary_df[summary_df["Generation Size"] == gen]
            print(f"\nAccuracy Summary for Generation Size = {gen}:")
            print(df_gen)
            
            # Set the font to "Times New Roman"
            plt.rcParams["font.family"] = "Times New Roman"
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Set positions for each group of bars
            bar_width = 0.35
            index = np.arange(len(df_gen))
            
            # Plot bars with updated colors
            bar1 = ax.bar(index, df_gen[bar1_name], bar_width, label=bar1_name, color='#1f77b4')  # Soft blue
            bar2 = ax.bar(index + bar_width, df_gen[bar2_name], bar_width, label=bar2_name, color='#ff7f0e')  # Soft orange
            
            # Customize plot
            ax.set_xlabel(x_label_title)
            ax.set_ylabel(y_label)
            ax.set_yscale('log')
            ax.set_title(f"{title}\nGeneration Size = {gen}", fontweight='bold', fontsize=14)
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(df_gen["Train Samples"])
            ax.legend()
            
            # Set the font for axis ticks
            ax.tick_params(axis="both", labelsize=12)
            
            plt.tight_layout()
    else:
        # Original behavior if no "Generation Size" column exists
        print("\nAccuracy Summary:")
        print(summary_df)
        
        # Set the font to "Times New Roman"
        plt.rcParams["font.family"] = "Times New Roman"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(summary_df))
        
        # Plot bars with updated colors
        bar1 = ax.bar(index, summary_df[bar1_name], bar_width, label=bar1_name, color='#1f77b4')
        bar2 = ax.bar(index + bar_width, summary_df[bar2_name], bar_width, label=bar2_name, color='#ff7f0e')
        
        # Customize plot
        ax.set_xlabel(x_label_title)
        ax.set_ylabel(y_label)
        ax.set_yscale('log')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(summary_df["Train Samples"])
        ax.legend()
        
        ax.tick_params(axis="both", labelsize=12)
        
        plt.tight_layout()

    plt.savefig(file_name, dpi=300)
    plt.show()

def normal_plot(
        sample_sizes,
        accuracy_before,
        accuracy_after,    
        title_before = "Real Only",
        title_after = "After Concatenation",
        x_label_title = 'Training Size',
        y_label_title = "Accuracy",
        title = "KNN Accuracy vs. Training Size (Real vs. Augmented)"
    ):
    plt.figure(figsize=(8, 5))
    plt.plot(sample_sizes, [accuracy_before[s] for s in sample_sizes], marker='o', linestyle='-', color='b', markersize=8, label=title_before)
    plt.plot(sample_sizes, [accuracy_after[s] for s in sample_sizes], marker='s', linestyle='--', color='r', markersize=8, label=title_after)
    plt.xlabel(x_label_title , fontsize=14, fontfamily="Times New Roman")
    plt.ylabel(y_label_title, fontsize=14, fontfamily="Times New Roman")
    plt.title(title, fontsize=16, fontfamily="Times New Roman")
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

def normal_plot_dict(
        sample_sizes,
        accuracy_dict: dict,
        x_label_title = 'Training Size',
        y_label_title = "Accuracy",
        title = "KNN Accuracy vs. Training Size (Real vs. Augmented)"
    ):
    '''
    accuracy_dict {name for the plot: values}
    '''
    plt.figure(figsize=(8, 5))
    for val_title, values in accuracy_dict.items():
        plt.plot(sample_sizes, [values[s] for s in sample_sizes], marker='o', linestyle='-', color='b', markersize=8, label=val_title)
    plt.xlabel(x_label_title , fontsize=14, fontfamily="Times New Roman")
    plt.ylabel(y_label_title, fontsize=14, fontfamily="Times New Roman")
    plt.title(title, fontsize=16, fontfamily="Times New Roman")
    plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()