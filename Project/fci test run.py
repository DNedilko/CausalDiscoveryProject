import pandas as pd
import numpy as np
import os
from causallearn.search.ConstraintBased.FCI import fci
import json
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz, gsq, kci, chisq, mv_fisherz
from causallearn.search.ScoreBased.GES import ges
from typing import List, Tuple, Union, Optional
from utilities.utils import *
import itertools




def data_loader(retain: List[str], sex: bool = False, region: str = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Loads and preprocesses the HBSC dataset.

    :param retain: List of column names to retain from the dataset.
    :param sex: If True, returns separate DataFrames for females and males.
    :return: Cleaned DataFrame, or tuple of (female_df, male_df) if sex=True.
    """
    # Load dtype dictionary from JSON
    with open("dtype_dict.json", "r") as f:
        dtype_dict = json.load(f)

    # Read the CSV, keeping only the desired columns
    data = pd.read_csv(
        "HBSC/HBSC2018OAed1.1.csv",
        delimiter=";",
        dtype=dtype_dict,
        usecols=retain + ["agecat", "sex", "region"] if sex else retain + ["agecat", "region"],
        decimal=",",
        skipinitialspace=True
    )

    # data = data[(data["agecat"].notna())][data["region"] == "GB-WLS"]
    data = data[(data["agecat"].notna())]
    if region is not None:
        data = data[data["region"] == region]
    print(data)

    for col in retain:
        if data[col].dtype == 'object' or pd.api.types.is_string_dtype(data[col]):
            data[col] = data[col].astype('category').cat.codes

    if sex:
        if "sex" not in data.columns:
            raise ValueError("'sex' column not found in the dataset.")


        data_fem = data[data["sex"] == 2][retain].reset_index(drop=True)
        data_men = data[data["sex"] == 1][retain].reset_index(drop=True)
        print(data_men.columns)
        return data_fem, data_men

    # Return only the retained columns
    return data[retain].reset_index(drop=True)

def data_prep(data: pd.DataFrame, how: str = 'all') -> np.ndarray:
    '''
    Cleans a DataFrame for causal discovery.

    :param data: DataFrame to be cleaned
    :param how: 'any' to drop rows with at least one NaN, 'all' to drop rows where all entries are NaN
    :return: Clean numpy array
    '''
    data_clean = data.dropna(how=how)
    non_numerical_cols = data_clean.select_dtypes(exclude=['number']).columns.tolist()

    try:
        data_clean = data_clean.fillna(0).astype(np.int32)
    except ValueError:
        raise ValueError("Non-numeric columns detected. Please preprocess your data accordingly.")

    data_array = data_clean.values

    return data_array


test_map = {
        "chisq": chisq,
        "gsq": gsq,
        #"fisherz": fisherz,
        #"mv_fisherz": mv_fisherz,
        #"kci": kci
}



@timeit
def run_fci(
        data: np.ndarray,
        test_map: dict,
        citest: str = "gsq",
        sign_level: float = 0.05,
        subscript: str = None,
        feature_names: Optional[list] = None,
        verbose: bool = False
) -> None:
    """
    Runs FCI algorithm and saves the resulting PAG.

    :param data: Input data array (n_samples x n_features)
    :param test_map: Dictionary mapping test names to conditional independence tests
    :param citest: Name of conditional independence test (must be a key in test_map)
    :param sign_level: Significance level for conditional independence tests
    :param subscript: Additional identifier for output filename
    :param feature_names: List of feature names for graph labels
    """
    # Validate inputs
    if citest not in test_map:
        raise ValueError(f"Invalid test '{citest}'. Available tests: {list(test_map.keys())}")

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(data.shape[1])]
    elif len(feature_names) != data.shape[1]:
        raise ValueError("feature_names length must match data columns")

    output_dir = "Graphs_loop_FCI"
    os.makedirs(output_dir, exist_ok=True)

    g, pag = fci(data, test_map.get(citest), alpha=sign_level, verbose=verbose)

    filename = f"FCI_{citest}_{sign_level}_{subscript}.png"
    output_path = os.path.join(output_dir, filename)

    # Save graph visualization
    GraphUtils.to_pydot(g, labels=feature_names).write_png(
        output_path
    )

    caption = f"Findings from the FCI algorithm with {citest}, performed at a significance level of {sign_level}{f' for {subscript}' if subscript is not None else ''}."
    print("The caption is ", caption)
    label = f"fci_{citest}_{sign_level}{subscript}"
    picture_to_tex(output_path, caption=caption, label= label)
    print("Tex wrapper for the graph is rendered")
    return label




# def run_ges(data,subscript = "", score_func = "", ):
#     os.makedirs("Graphs_loop_FCI", exist_ok=True)
#     # default parameters
#     #Record = ges(data)
#     # # or customized parameters
#     Record = ges(data,
#                  score_func,
#                  maxP,
#                  parameters)
#
#     filename = f"GES_{score_func}_{maxP}_{subscript}.png"
#     pyd = GraphUtils.to_pydot(Record['G'], labels=data.columns.tolist()).write_png(
#         os.path.join("Graphs_loop_GES", filename)
#     )
#     print("Done for ", filename)
@timeit
def iterator_over_fci(retain, how : str = "all", verbose: bool  = False, region: str = "UA"):
    """
    Runs the FCI algorithm for each CI test and significance level, recording execution time and results.

    Parameters:
        retain (list): List of feature names to retain for analysis.

    Returns:
        pd.DataFrame: DataFrame with columns ["CI test", "alpha", "time", "result"] summarizing the runs.
    """
    # Load and preprocess data
    if verbose:
        print("Printing out the status")
    data = data_loader(retain, region = region)
    print(data.info())
    # data_fem, data_men = data_loader(retain, sex=True)
    data_preped = data_prep(data, "all")
    data_list = list(map(data_prep, [
        # data_fem, data_men,
        data], itertools.repeat(how)))
    print(f"Data shape after cleaning: {list(map(len, data_list))}")
    groups = [
        # "female", "male",
               "all"]
    data_dictionary = dict(zip(groups, data_list))
    # Prepare results DataFrame
    results = pd.DataFrame(columns=["CI test", "alpha", "time", "result"])
    significance_levels = [0.001,
                           0.005,
                           0.01,
                           0.025,
                           0.05,
                           0.075,
                           0.1
                           ]


    #Iterate over CI tests and significance levels
    for gr in data_dictionary.keys():
        for ci_test in test_map.keys():
            for alpha in significance_levels:
                print(f"Running FCI with {ci_test} at alpha={alpha}")
                start_time = time.time()
                fci_data = data_dictionary[gr]
                label = run_fci(
                    fci_data,
                    test_map,
                    ci_test,
                    sign_level=alpha,
                    feature_names=retain,
                    subscript = f'{gr}_{region}_{len(retain)}_{retain[-1]}',
                    verbose = verbose
                )
                elapsed_time = time.time() - start_time
                ref = f"\\ref{{fig:{label}}}"
                results.loc[len(results)] = [ci_test, alpha, elapsed_time, ref]
    label = f"fci_parameters_time_{region}_{len(retain)}_{retain[-1]}"
    [length] = list(map(len, data_list))
    caption = f'Summary of optimal parameters search for FCI algorithm with corresponding time required for execution. The experiments are ran on {len(retain)} variables, sample size {length} records from {region}.'
    df_to_tex(results, caption=caption, label=label)
# for ci_test in test_map.keys():
#     for alpha in significance_levels:
#                 print(f"Running FCI with {ci_test} at alpha={alpha}")
#                 start_time = time.time()
#                 label = run_fci(
#                     data_preped,
#                     test_map,
#                     ci_test,
#                     sign_level=alpha,
#                     feature_names=retain,
#                     subscript = "all test 100 UA",
#                     verbose = verbose
#                 )
#                 elapsed_time = time.time() - start_time
#                 ref = f"\\ref{{fig:{label}}}"
        #             results.loc[len(results)] = [ci_test, alpha, elapsed_time, ref]





import itertools
import multiprocessing as mp
import pandas as pd
import time

def run_fci_wrapper(args):
    gr, data, ci_test, alpha, retain, verbose = args
    start_time = time.time()
    label = run_fci(
        data,
        test_map,
        ci_test,
        sign_level=alpha,
        feature_names=retain,
        subscript=gr,
        verbose=verbose
    )
    elapsed_time = time.time() - start_time
    ref = f"\\ref{{fig:{label}}}"

    return (gr, ci_test, alpha, elapsed_time, ref)

#@timeit
# def iterator_over_fci(retain, how: str = "all", verbose: bool = False):
#     """
#     Runs the FCI algorithm in parallel for each CI test and significance level, recording execution time and results.
#     """
#
#
#     # Load and preprocess data
#     data = data_loader(retain)
#     data_fem, data_men = data_loader(retain, sex=True)
#     data_list = list(map(data_prep, [data_fem, data_men, data], itertools.repeat(how)))
#     groups = ["female",
#               "male", "all"]
#     data_dictionary = dict(zip(groups, data_list))
#     print(f"Data shape after cleaning: {list(map(len, data_list))}")
#     output_dir = "Graphs_loop_FCI"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Prepare list of all jobs to parallelize
#     significance_levels = [0.1,
#                            0.05, 0.01, 0.001
#                            ]
#     tasks = [
#         (gr, data_dictionary[gr], ci_test, alpha, retain, verbose)
#         for gr in data_dictionary
#         for ci_test in test_map
#         for alpha in significance_levels
#     ]
#
#     # Run in parallel
#     with mp.Pool(mp.cpu_count()) as pool:
#         results_list = pool.map(run_fci_wrapper, tasks)
#
#     # Build DataFrame
#     results = pd.DataFrame(results_list, columns=["Group", "CI test", "alpha", "time", "result"])
#     caption = "FCI execution time for various ci tests and significance levels"
#     label = "fci_runs_alphas_ci_tests_datagroups"
#     df_to_tex(results, caption, label)
#     return results

import time
import pandas as pd
import matplotlib.pyplot as plt
import itertools




def plot_fci_runtimes(results, alpha_fixed=0.05, sample_size_fixed=2500):
    """
    Plots FCI runtime as a function of significance level and sample size for different CI tests.

    Parameters:
        results (pd.DataFrame): DataFrame with columns ["CI test", "alpha", "time", "sample size"].
        alpha_fixed (float): Alpha level to fix when plotting runtime vs sample size.
        sample_size_fixed (int): Sample size to fix when plotting runtime vs alpha.
    """

    # --- Plot 1: Runtime vs Significance Level (fixed sample size) ---
    plt.figure(figsize=(10, 6))
    for ci_test in results["CI test"].unique():
        df_plot = results[(results["CI test"] == ci_test) &
                          (results["sample size"] == sample_size_fixed)].sort_values("alpha")
        plt.plot(df_plot["alpha"], df_plot["time"], marker='o', label=ci_test)

    plt.xlabel("Significance Level (?)")
    plt.ylabel("Computation Time (s)")
    plt.title(f"FCI Runtime vs Significance Level (Sample Size = {sample_size_fixed})")
    plt.legend(title="CI Test")
    plt.grid(True)
    plt.tight_layout()
    name_alpha = "fci_runtime_vs_alpha"
    caption_alpha = (
        f"Computation time of the FCI algorithm across different significance levels ? "
        f"for various conditional independence (CI) tests. Results are based on a fixed sample size of {sample_size_fixed} observations."
    )
    plt.savefig(f"{name_alpha}.png", dpi=300)
    picture_to_tex(f"{name_alpha}.png", caption=caption_alpha, label=name_alpha)
    plt.show()

    # --- Plot 2: Runtime vs Sample Size (fixed alpha) ---
    plt.figure(figsize=(10, 6))
    for ci_test in results["CI test"].unique():
        df_plot = results[(results["CI test"] == ci_test) &
                          (results["alpha"] == alpha_fixed)].sort_values("sample size")
        plt.plot(df_plot["sample size"], df_plot["time"], marker='o', label=ci_test)

    plt.xlabel("Sample Size")
    plt.ylabel("Computation Time (s)")
    plt.title(f"FCI Runtime vs Sample Size (Significance Level = {alpha_fixed})")
    plt.legend(title="CI Test")
    plt.grid(True)
    plt.tight_layout()
    name_size = "fci_runtime_vs_samplesize"
    caption_size = (
        f"Computation time of the FCI algorithm as a function of sample size for different CI tests. "
        f"The significance level ? is fixed at {alpha_fixed} to isolate the effect of increasing data volume on performance."
    )
    plt.savefig(f"{name_size}.png", dpi=300)
    picture_to_tex(f"{name_size}.png", caption=caption_size, label=name_size)
    plt.show()


@timeit
def time_over_fci(retain, how: str = "all", verbose: bool = False, region: str = "UA"):
    """
    Runs the FCI algorithm for each CI test and significance level, recording execution time and results.
    Also plots computation time as a function of significance level (alpha).

    Parameters:
        retain (list): List of feature names to retain for analysis.

    Returns:
        pd.DataFrame: DataFrame with columns ["CI test", "alpha", "time", "result"] summarizing the runs.
    """
    if verbose:
        print("Loading and preprocessing data...")
    data = data_loader(retain, region=region)
    data_preped = data_prep(data, "all")
    data_list = list(map(data_prep, [data], itertools.repeat(how)))
    groups = ["all"]
    data_dictionary = dict(zip(groups, data_list))

    results = pd.DataFrame(columns=["CI test", "alpha", "time", "sample size"])
    significance_levels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1]

    # Store timing for plotting
    time_records = {ci_test: [] for ci_test in test_map.keys()}

    for gr in data_dictionary:
        for ci_test in test_map.keys():
            for alpha in significance_levels:
                for size in range(250, 3000, 250):
                    print(f"Running FCI with {ci_test} at alpha={alpha}")
                    start_time = time.time()
                    np.random.seed(42)
                    indices = np.random.choice(data_dictionary[gr].shape[0], size=size, replace=False)
                    fci_data = data_dictionary[gr][indices]
                    label = run_fci(
                        fci_data,
                        test_map,
                        ci_test,
                        sign_level=alpha,
                        feature_names=retain,
                        subscript=f'_for time experiment',
                        verbose=verbose
                    )
                    elapsed_time = time.time() - start_time
                    ref = f"\\ref{{fig:{label}}}"
                    results.loc[len(results)] = [ci_test, alpha, elapsed_time, size]
                    time_records[ci_test].append(elapsed_time)

    # Save table as LaTeX
    label = f"fci_parameters_time_experiments"
    [length] = list(map(len, data_list))
    caption = (
        f'Summary of execution times for the FCI algorithm across varying sample sizes, conditional independence (CI) tests, and significance levels (\(\alpha\)). Each row corresponds to a unique combination of parameters used to run the algorithm. The results illustrate how computational time is affected by different test types, confidence thresholds, and data sizes.'
    )
    df_to_tex(results, caption=caption, label=label)

    plot_fci_runtimes(results)

    return results



if __name__ == "__main__":
    retain = ["bulliedothers", "beenbullied",
              "cbulliedothers", "cbeenbullied",
              "fight12m",
              # "injured12m",
              "lifesat", "famhelp", "famsup", "famtalk",
              "famdec", "friendhelp", "friendcounton", "friendshare", "friendtalk",
              "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
              "teacheraccept", "teachercare", "teachertrust","timeexe", "talkfather", "talkstepfa", "talkstepmo", "talkmother"
              ]
    print(retain)
    # retain = ["sex", "agecat",
    #           # "IRFAS",
    #           # "IRRELFAS_LMH",
    #           # "IOTF4",
    #
    #           "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7",
    #           "emcsocmed8", "emcsocmed9", "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
    #           "teacheraccept", "teachercare", "teachertrust", "bulliedothers", "beenbullied", "cbulliedothers",
    #           "cbeenbullied", "fight12m", "injured12m", "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
    #           "emconlpref1", "emconlpref2", "emconlpref3", "friendhelp", "friendcounton", "friendshare", "friendtalk",
    #           "famhelp", "famsup", "famtalk", "famdec",
    #           "talkfather", "talkstepfa", "talkmother", "talkstepmo", "timeexe", "health", "lifesat", "headache",
    #           "stomachache", "backache", "feellow", "irritable", "nervous",
    #           "sleepdificulty", "dizzy", "thinkbody", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1",
    #           "fosterhome1", "elsehome1_2",
    #           "employfa", "employmo", "employnotfa", "employnotmo", "fasfamcar", "fasbedroom", "fascomputers",
    #           "fasbathroom", "fasdishwash", "fasholidays", "physact60", "bodyweight", "bodyheight"]
    #

    new_retain = [
    # Demographics & Metadata
    "agecat", "sex",
    # Family Affluence Scale
    "fasfamcar", "fasbedroom", "fascomputers", "fasbathroom", "fasdishwash", "fasholidays",
    # Health & Well-being
    "health", "lifesat", "thinkbody", "headache", "stomachache", "backache", "feellow", "irritable", "nervous", "sleepdificulty", "dizzy",
    # Health Behaviors
    "physact60", "timeexe",
    # Body Measures
    "bodyweight", "bodyheight",
    # School Experience
    "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept", "teacheraccept", "teachercare", "teachertrust",
    # Violence and Bullying
    "bulliedothers", "cbulliedothers", "fight12m", "injured12m", "beenbullied", "cbeenbullied",
    # Peer Support
    "friendhelp", "friendcounton", "friendshare", "friendtalk",
    # Emotional Communication Preferences
    "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
    "emconlpref1", "emconlpref2", "emconlpref3",
    "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9",
    # Migration Background
    "countryborn", "countrybornmo", "countrybornfa",
    # Household Composition
    "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2",
    # Parental Employment
    "employfa", "employmo", "employnotfa", "employnotmo",
    # Parent--Child Communication
    "talkfather", "talkstepfa", "talkmother", "talkstepmo",
    # Family Support
    "famhelp", "famsup", "famtalk", "famdec"
]

    time_over_fci(retain, "all", False, region="UA")
    # iterator_over_fci(new_retain, "all", False, region = "UA")
    #iterator_over_fci(retain, how='any', verbose=False)
    # retain = [
    #     # "sex", "agecat", "IRFAS",
    #     # "IRRELFAS_LMH",
    #     #       "IOTF4",
    #     #       "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9",
    #     #       "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
    #     #     "teacheraccept", "teachercare", "teachertrust",
    #     #     "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m",
    #     #     "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
    #     #     "emconlpref1", "emconlpref2", "emconlpref3",
    #     #     "friendhelp", "friendcounton", "friendshare", "friendtalk", "famhelp", "famsup", "famtalk", "famdec",
    #     #     "talkfather", "talkstepfa", "talkmother", "talkstepmo", "timeexe",        "health", "lifesat", "headache", "stomachache", "backache", "feellow", "irritable", "nervous",
    #     "sleepdificulty", "dizzy", "thinkbody", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1",
    #     "fosterhome1", "elsehome1_2",
    #     # "employfa", "employmo", "employnotfa", "employnotmo", "fasfamcar",    "fasbedroom",    "fascomputers",    "fasbathroom",    "fasdishwash",    "fasholidays"
    # ]





