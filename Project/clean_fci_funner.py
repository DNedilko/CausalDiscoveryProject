import pandas as pd
import time
from cdt.causality.graph import FCI
import networkx as nx
import matplotlib.pyplot as plt
import os

def load_and_prepare_data(filepath, columns, drop_na=True):
    """
    Load and preprocess the dataset.

    Parameters:
        filepath (str): Path to the CSV file.
        columns (list): List of columns to retain.
        drop_na (bool): Whether to drop rows with missing values.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = pd.read_csv(filepath, usecols=columns)
    if drop_na:
        df = df.dropna()
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def run_fci(data, ci_test='gsq', alpha=0.05):
    """
    Run the FCI algorithm on the data.

    Parameters:
        data (pd.DataFrame): The input data.
        ci_test (str): Conditional independence test ('gsq', etc.).
        alpha (float): Significance level.

    Returns:
        networkx.Graph: The inferred PAG.
    """
    fci = FCI(ci_test=ci_test, alpha=alpha)
    output_graph = fci.predict(data)
    return output_graph

def visualize_graph(graph, output_path):
    """
    Visualize and save the causal graph.

    Parameters:
        graph (networkx.Graph): The graph to visualize.
        output_path (str): Path to save the PNG image.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def run_pipeline(filepath, columns, ci_tests, alphas, output_dir):
    """
    Orchestrate the causal discovery pipeline over multiple configurations.

    Parameters:
        filepath (str): Path to the dataset.
        columns (list): Columns to use.
        ci_tests (list): List of CI tests to use.
        alphas (list): List of significance levels.
        output_dir (str): Directory to save results.

    Returns:
        pd.DataFrame: Summary of results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = []
    data = load_and_prepare_data(filepath, columns)
    for ci_test in ci_tests:
        for alpha in alphas:
            start_time = time.time()
            graph = run_fci(data, ci_test=ci_test, alpha=alpha)
            duration = time.time() - start_time
            graph_path = f"{output_dir}/fci_{ci_test}_alpha{alpha}.png"
            visualize_graph(graph, graph_path)
            results.append({
                'ci_test': ci_test,
                'alpha': alpha,
                'runtime_sec': duration,
                'graph_path': graph_path
            })
    return pd.DataFrame(results)

# Example usage:
if __name__ == "__main__":
    FILEPATH = "hbsc_data.csv"
    COLUMNS = ['age', 'var1', 'var2', 'var3']  # replace with actual variable names
    CI_TESTS = ['gsq']
    ALPHAS = [0.01, 0.05]
    OUTPUT_DIR = "./results"

    results_df = run_pipeline(FILEPATH, COLUMNS, CI_TESTS, ALPHAS, OUTPUT_DIR)
    results_df.to_csv(f"{OUTPUT_DIR}/fci_results_summary.csv", index=False)
