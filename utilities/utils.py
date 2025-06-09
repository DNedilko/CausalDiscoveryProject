import os
import pandas as pd
import shutil

def df_to_tex(
    df: pd.DataFrame,
    caption: str = "Random table",
    label: str = "random_table",
    path: str = "C:/Users/dnedi/PycharmProjects/CausalDiscoveryProject/Report/final_report/tables/"
) -> None:
    """
    Export a pandas DataFrame to a LaTeX file with caption and label.

    :param df: The DataFrame to export.
    :param caption: Caption for the table.
    :param label: Label for the table (used in LaTeX and as filename).
    :param path: Directory to store the .tex file.
    :return: None
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    # Create LaTeX string
    latex_str = df.to_latex(
        caption=caption,
        label=f'tab:{label}',
        float_format="%.3f".__mod__,
        longtable=False,
        multicolumn_format='c',
        multirow=True,
        index=False  # Optionally suppress index column
    )
    # lines = latex_str.splitlines()
    # caption_line = [line for line in lines if line.strip().startswith(r'\caption')]
    # if caption_line:
    #     lines.remove(caption_line[0])
    #     # Find where to insert: after \begin{longtable} or \begin{tabular}
    #     for i, line in enumerate(lines):
    #         if line.strip().startswith(r'\begin{longtable}') or line.strip().startswith(r'\begin{tabular}'):
    #             insert_at = i + 1
    #             break
    #     else:
    #         insert_at = 1  # fallback
    #     lines.insert(insert_at, caption_line[0])
    #     latex_str = "\n".join(lines)
    tex_filename = f"{label}.tex"
    tex_file_path = os.path.join(path, tex_filename)
    with open(tex_file_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"LaTeX table with caption below saved to {tex_file_path}")


def picture_to_tex(
        image_src_path: str,
        caption: str = '',
        label: str = '',
        path: str = "C:/Users/dnedi/PycharmProjects/CausalDiscoveryProject/Report/final_report/pictures/",
        width: str = '0.8\\textwidth'
) -> None:
    """
    Moves an image to the specified directory and saves a LaTeX file to include it with caption and label.

    :param image_src_path: Path to the source image file
    :param caption: Caption text for the figure
    :param label: Label for referencing the figure
    :param path: Directory to save the image and .tex file
    :param width: Width of the image in LaTeX units (default 0.8\textwidth)
    :return: None
    """
    os.makedirs(path, exist_ok=True)
    image_filename = os.path.basename(image_src_path)
    image_dst_path = os.path.join(path, image_filename)

    # Move the image file
    shutil.move(image_src_path, image_dst_path)

    # Generate LaTeX code
    latex_code = f"""\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width={width}]{{Report/final_report/pictures/{image_filename}}}
    \\caption{{{caption}}}
    \\label{{fig:{label}}}
\\end{{figure}}
"""
    # Save LaTeX code to file
    tex_filename = f"{label}.tex"
    tex_file_path = os.path.join(path, tex_filename)
    with open(tex_file_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"Image moved to: {image_dst_path}")
    print(f"LaTeX code saved to: {tex_file_path}")


from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function {func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Example usage:
# move_picture_and_save_tex('C:/somewhere/myplot.png', 'My plot caption', 'my_plot_label')
