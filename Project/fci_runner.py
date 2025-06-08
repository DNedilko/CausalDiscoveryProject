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




def data_loader(retain: List[str], sex: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
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
    data_clean = data.dropna(how=how).fillna(0)

    try:
        data_clean = data_clean.astype(np.int32)
    except ValueError:
        raise ValueError("Non-numeric columns detected. Please preprocess your data accordingly.")

    data_array = data_clean.values

    return data_array

def data_prep_leave_none(data):
    # Drop rows with missing values in retained columns
    data_clean = data.dropna( how='all').reset_index(drop=True)

    # Convert all columns to integers if appropriate (or to categorical)
    data_clean = data_clean

    # Convert DataFrame to numpy array for FCI
    data_array = data_clean.values

    return  data_array

test_map = {
        #"chisq": chisq,
        "gsq": gsq,
        #"fisherz": fisherz,
        #"mv_fisherz": mv_fisherz,

        # "kci": kci
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
        output_path)

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
#@timeit
# def iterator_over_fci(retain, how : str = "all", verbose: bool  = False):
#     """
#     Runs the FCI algorithm for each CI test and significance level, recording execution time and results.
#
#     Parameters:
#         retain (list): List of feature names to retain for analysis.
#
#     Returns:
#         pd.DataFrame: DataFrame with columns ["CI test", "alpha", "time", "result"] summarizing the runs.
#     """
#     # Load and preprocess data
#     if verbose:
#         print("Printing out the status")
#     data = data_loader(retain)
#     data_fem, data_men = data_loader(retain, sex=True)
#     data_list  = list(map(data_prep, [data_fem, data_men, data], itertools.repeat(how)))
#     print(f"Data shape after cleaning: {list(map(len, data_list))}")
#     groups = ["female", "male", "all"]
#     data_dictionary = dict(zip(groups, data_list))
#
#     # Prepare results DataFrame
#     results = pd.DataFrame(columns=["CI test", "alpha", "time", "result"])
#     significance_levels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1][::-1]
#
#
#     # Iterate over CI tests and significance levels
#     for gr in data_dictionary.keys():
#         for ci_test in test_map.keys():
#             for alpha in significance_levels:
#                 print(f"Running FCI with {ci_test} at alpha={alpha}")
#                 start_time = time.time()
#                 label = run_fci(
#                     data_dictionary[gr],
#                     test_map,
#                     ci_test,
#                     sign_level=alpha,
#                     feature_names=retain,
#                     subscript = gr,
#                     verbose = verbose
#                 )
#                 elapsed_time = time.time() - start_time
#                 ref = f"\\ref{{fig:{label}}}"
#                 results.loc[len(results)] = [ci_test, alpha, elapsed_time, ref]
#
#     return results


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

@timeit
def iterator_over_fci(retain, how: str = "all", verbose: bool = False):
    """
    Runs the FCI algorithm in parallel for each CI test and significance level, recording execution time and results.
    """


    # Load and preprocess data
    data = data_loader(retain)
    data_fem, data_men = data_loader(retain, sex=True)
    data_list = list(map(data_prep, [data_fem, data_men, data], itertools.repeat(how)))
    groups = ["female", "male", "all"]
    data_dictionary = dict(zip(groups, data_list))
    print(f"Data shape after cleaning: {list(map(len, data_list))}")
    output_dir = "Graphs_loop_FCI"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare list of all jobs to parallelize
    significance_levels = [0.1, 0.05, 0.01, 0.001]
    tasks = [
        (gr, data_dictionary[gr], ci_test, alpha, retain, verbose)
        for gr in data_dictionary
        for ci_test in test_map
        for alpha in significance_levels
    ]

    # Run in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        results_list = pool.map(run_fci_wrapper, tasks)

    # Build DataFrame
    results = pd.DataFrame(results_list, columns=["Group", "CI test", "alpha", "time", "result"])
    caption = "FCI execution time for various ci tests and significance levels"
    label = "fci_runs_alphas_ci_tests_datagroups"
    df_to_tex(results, caption, label)
    return results



if __name__ == "__main__":
    retain = ["bulliedothers", "beenbullied",
              "cbulliedothers", "cbeenbullied",
              "fight12m",
              "injured12m",
              "lifesat", "famhelp", "famsup", "famtalk",
              "famdec", "friendhelp", "friendcounton", "friendshare", "friendtalk",
              "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
              "teacheraccept", "teachercare", "teachertrust", "IRRELFAS_LMH"
              ]


    iterator_over_fci(retain, "any", False)
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





