# import os
# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.0"  # Replace with your R path
# os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.5.0\bin\x64"
#
# # Now import rpy2
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# import rpy2.situation
# for row in rpy2.situation.iter_info():
#     print(row)

#
# import os
# print(os.environ.get("R_HOME"))
# print(os.environ["PATH"])
# import platform
# print(platform.architecture())
#
# import rpy2.robjects as ro
# print(ro.r('version'))
#
# import os
# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.0"
# os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.5.0\bin\x64"
# import os
# os.environ["RPY2_CFFI_MODE"] = "ABI"
#
# os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.0"
# os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.5.0\bin\x64"
#
# import rpy2.robjects as ro
# print(ro.r('version'))
#
#
# import rpy2
# print(rpy2)
# print(rpy2.__file__)
#
# import sys
# print(sys.executable)

# import rpy2
# print(rpy2.__version__)
# import rpy2.robjects as ro
# print(ro.r('version'))
# import sys
# print(sys.executable)
# import rpy2
# print(rpy2)
# print(getattr(rpy2, '__version__', 'no version'))
#
# import rpy2
# print(rpy2)
# print(getattr(rpy2, '__version__', 'no version'))
# import rpy2.robjects as ro
# print(ro)
#
import numpy as np
import time
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq


def generate_hbsc_causal_data(n_samples, n_vars=20, n_categories=5):
    """Generate HBSC-like data with embedded causal relationships"""
    data = pd.DataFrame()

    # Core demographics (fixed distributions)
    data['age'] = np.random.choice([11, 13, 15], n_samples, p=[0.33, 0.34, 0.33])
    data['sex'] = np.random.choice([1, 2], n_samples, p=[0.493, 0.507])

    # Family Affluence Scale III (FAS III) with causal structure
    fas_features = np.zeros((n_samples, 5))
    for i in range(5):
        if i == 0:
            fas_features[:, i] = np.random.choice([1, 2, 3], n_samples, p=[0.12, 0.37, 0.51])
        else:
            for k in range(3):
                mask = fas_features[:, i - 1] == k + 1
                prob = np.array([0.1, 0.3, 0.6]) if k == 0 else \
                    np.array([0.2, 0.5, 0.3]) if k == 1 else \
                        np.array([0.05, 0.15, 0.8])
                fas_features[mask, i] = np.random.choice([1, 2, 3], np.sum(mask), p=prob)

    # Health outcomes with multi-parent dependencies
    health_data = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        if i < 5:  # First 5 variables are FAS III components
            health_data[:, i] = fas_features[:, i]
        else:
            # Dual parent interactions with FAS components
            for prev1 in range(3):
                for prev2 in range(3):
                    mask = (health_data[:, i - 1] == prev1) & (health_data[:, i - 5] == prev2)
                    if mask.sum() == 0:
                        continue

                    # Health outcome probabilities based on FAS and previous health
                    base_prob = 0.6 if prev2 == 3 else 0.4  # Higher FAS better health
                    prob = np.full(n_categories, (1 - base_prob) / (n_categories - 1))
                    target = (prev1 + prev2) % n_categories
                    prob[target] = base_prob
                    prob /= prob.sum()

                    health_data[mask, i] = np.random.choice(n_categories, np.sum(mask), p=prob)

    # Add HBSC-specific distributions and missing data
    data = pd.concat([data, pd.DataFrame(health_data)], axis=1)

    # Introduce missing values (25% in key variables)
    missing_mask = np.random.rand(n_samples, 5) < 0.25
    for i in range(5):
        data.loc[missing_mask[:, i], f'var_{i + 5}'] = np.nan

    return data.values.astype(int)


def run_fci_hbsc_analysis(var_counts, alpha=0.05, n_samples=10000):
    """Run causal discovery on HBSC-like data"""
    results = []

    for n_vars in var_counts:
        print(f"\n🔍 Analyzing {n_vars} HBSC-style variables...")

        # Generate data with causal structure
        data = generate_hbsc_causal_data(n_samples, n_vars)

        # Run FCI algorithm
        start_time = time.time()
        _, pag = fci(data, gsq, alpha=alpha, verbose=False)
        elapsed_time = time.time() - start_time

        # Store results
        results.append({
            'variables': n_vars,
            'time': elapsed_time,
            'edges': [str(e) for e in pag]
        })

        print(f"⏳ Processing time: {elapsed_time:.2f}s")
        print(f"🕸️  Detected {len(pag)} causal relationships")

    return results


# Example analysis pipeline
analysis_results = run_fci_hbsc_analysis(
    var_counts=[10, 20, 30],
    n_samples=5000,
    alpha=0.01
)


# Generate sample dataset
hbsc_data = generate_hbsc_causal_data(100000, n_vars=50)

# Run full analysis
results = run_fci_hbsc_analysis([10, 15, 20,50  ], n_samples=5000)

# Inspect results
print(f"Average edges for 20 variables: {len(results[2]['edges'])}")
