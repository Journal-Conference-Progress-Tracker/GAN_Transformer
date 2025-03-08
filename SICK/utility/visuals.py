
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
def save_table_as_image(
        summary_df,
        file_name="accuracy_summary.png",
        title="Accuracy Comparisons: Real Only v.s. After Concatenation",
        decimal_places=2,
        max_col_width=20  # Maximum character width before wrapping
    ):
    """
    Save the entire DataFrame as an image with properly wrapped headers.
    - Wraps headers at spaces for better readability
    - Rounds numerical values to `decimal_places`
    - Adjusts row height dynamically based on header size
    """
    # Round numerical values to the specified decimal places
    summary_df = summary_df.copy()
    for col in summary_df.select_dtypes(include=[np.number]).columns:
        summary_df[col] = summary_df[col].apply(lambda x: f"{x:.{decimal_places}f}")

    # Convert DataFrame to a list of lists for table display
    summary_data = summary_df.values.tolist()
    def wrap_header(header, width):
        # Split the header into words
        words = header.split(' ')
        
        # If the header is shorter than the max width, no wrapping is needed.
        if len(header) <= width:
            return header

        lines = []
        current_line = ""
        
        # Iterate through each word and build lines that do not exceed the max width.
        for word in words:
            # Check if adding the next word exceeds the width (account for a space if needed)
            if len(current_line) + len(word) + (1 if current_line else 0) > width:
                lines.append(current_line)
                current_line = word
            else:
                # Add a space before the word if current_line is not empty
                current_line = f"{current_line} {word}".strip()
        
        # Append the final line if it exists
        if current_line:
            lines.append(current_line)
        
        # Join the lines with newline characters for display
        return "\n".join(lines)


    wrapped_headers = [wrap_header(col, max_col_width) for col in summary_df.columns]

    # Determine the maximum number of wrapped lines in any header
    max_header_lines = max(len(header.split("\n")) for header in wrapped_headers)
    # Define a header height factor (in inches) based on the number of lines.
    # (Adjust the multiplier as needed based on your font size.)
    header_height = 0.05 * max_header_lines

    # Calculate figure dimensions based on data and header size
    num_rows, num_cols = summary_df.shape
    fig_width = max(10, num_cols * 1.5)
    # Add extra height for the header rows
    fig_height = max(6, num_rows * 0.5 + header_height + 0.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=summary_data,
        colLabels=wrapped_headers,
        loc="center",
        cellLoc="center",
        colWidths=[max_col_width * 0.06 for _ in wrapped_headers]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Scale the table for readability

    # Adjust the height of header cells (row 0) to match the header text
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(header_height)

    # Set the table title
    ax.set_title(title, fontweight='bold', fontsize=16, family="Times New Roman")

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    print(f"Table has been saved as '{file_name}'")


def display_accuracy_summary_with_more_than_two_bars(
        summary_df,
        x_label_title="Train Samples",
        y_label="Accuracy",
        title="Accuracy Comparisons: Real Only v.s. After Concatenation",
        file_name="plot.png"
    ):

    print("\nAccuracy Summary:")
    print(summary_df)
    
    # Set font to "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    
    num_categories = len(summary_df)  # Number of rows (x-axis categories)
    num_bars = len(summary_df.columns) - 1  # Excluding "Train Samples"

    fig, ax = plt.subplots(figsize=(max(8, num_bars * 1.5), 6))  # Adjust figure size based on bars
    index = np.arange(num_categories)  # X-axis positions

    # Dynamically calculate bar width based on number of bars
    bar_width = max(0.1, min(0.8 / num_bars, 0.3))  # Ensures bars are well spaced

    bar_columns = [col for col in summary_df.columns if col != "Train Samples"]  # Exclude x-axis label

    # Plot bars with spacing adjustments
    for i, column in enumerate(bar_columns):
        ax.bar(index + i * bar_width, summary_df[column], bar_width, label=column)

    # Customize plot
    ax.set_xlabel(x_label_title)
    ax.set_ylabel(y_label)
    ax.set_yscale('log')
    ax.set_title(title, fontweight='bold', fontsize=14)
    
    # Center x-axis ticks
    ax.set_xticks(index + (num_bars - 1) * bar_width / 2)
    ax.set_xticklabels(summary_df["Train Samples"])
    
    ax.legend()
    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.show()


def display_accuracy_summary(
        summary_df,
        x_label_title="Train Samples",
        y_label="Accuracy",
        title="Accuracy Comparisons: Real Only v.s. After Concatenation",
        file_name="plot.png",
        bar1_name="Real Only Accuracy",
        bar2_name="After Concatenation Accuracy"
    ):

    print("\nAccuracy Summary:")
    print(summary_df)
    
    # Set the font to "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    
    fig, ax = plt.subplots(figsize=(20, 6))
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