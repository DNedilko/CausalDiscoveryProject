import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq
import json


retain = ["sex", "agecat", "IRFAS",
          #"IRRELFAS_LMH",
          "IOTF4",         "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9",         "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
        "teacheraccept", "teachercare", "teachertrust",        "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m",        "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
        "emconlpref1", "emconlpref2", "emconlpref3",        "friendhelp", "friendcounton", "friendshare", "friendtalk", "famhelp", "famsup", "famtalk", "famdec",
        "talkfather", "talkstepfa", "talkmother", "talkstepmo", "timeexe",        "health", "lifesat", "headache", "stomachache", "backache", "feellow", "irritable", "nervous",
        "sleepdificulty", "dizzy", "thinkbody","motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2",
        "employfa", "employmo", "employnotfa", "employnotmo", "fasfamcar",    "fasbedroom",    "fascomputers",    "fasbathroom",    "fasdishwash",    "fasholidays"]



with open("dtype_dict.json", "r") as f:
    dtype_str_dict = json.load(f)

# Convert strings back to numpy dtypes if needed
dtype_dict = {k: v for k, v in dtype_str_dict.items()}








data = pd.read_csv(
    "HBSC/HBSC2018OAed1.1.csv",
    delimiter=";",
    dtype=dtype_dict,
    usecols=retain,
    decimal=",",
    skipinitialspace=True
)

# Drop rows with missing values in retained columns
data_clean = data.dropna().reset_index(drop=True)

# Convert all columns to integers if appropriate (or to categorical)
data_clean = data_clean.astype(int)

# Convert DataFrame to numpy array for FCI
data_array = data_clean.values

# Run FCI algorithm with G-squared test for categorical data
g, pag = fci(data_array, gsq, alpha=0.05, verbose=True)

# Print discovered edges
print("Discovered PAG edges:")
for edge in pag:
    print(edge)


