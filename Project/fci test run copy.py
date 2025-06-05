import pandas as pd
import numpy as np
import os
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import gsq
import json
from causallearn.utils.GraphUtils import GraphUtils


def data_loader(retain):
    retain = [
        # "sex", "agecat", "IRFAS",
        #       #"IRRELFAS_LMH",
        #       "IOTF4",         "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9",         "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
        #     "teacheraccept", "teachercare", "teachertrust",        "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m",        "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
        #     "emconlpref1", "emconlpref2", "emconlpref3",        "friendhelp", "friendcounton", "friendshare", "friendtalk", "famhelp", "famsup", "famtalk", "famdec",
        #     "talkfather", "talkstepfa", "talkmother", "talkstepmo", "timeexe",        "health", "lifesat", "headache", "stomachache", "backache", "feellow", "irritable", "nervous",
        "sleepdificulty", "dizzy", "thinkbody", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1",
        "fosterhome1", "elsehome1_2",
        # "employfa", "employmo", "employnotfa", "employnotmo", "fasfamcar",    "fasbedroom",    "fascomputers",    "fasbathroom",    "fasdishwash",    "fasholidays"
    ]

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
    return data
def data_prep(data):
    # Drop rows with missing values in retained columns
    data_clean = data.dropna().reset_index(drop=True)

    # Convert all columns to integers if appropriate (or to categorical)
    data_clean = data_clean.astype(int)

    # Convert DataFrame to numpy array for FCI
    data_array = data_clean.values

    return  data_array

def run_fci(data_array):
    # Run FCI algorithm with G-squared test for categorical data
    g, pag = fci(data_array, gsq, alpha=0.05, verbose=True)

    # Print discovered edges
    print("Discovered PAG edges:")
    for edge in pag:
        print(edge)


    filename = f"FCI_gsq_alph0.05_testruncopy.png"
    GraphUtils.to_pydot(g, labels=data.columns.tolist()).write_png(
        os.path.join("Graphs_tets", filename)
    )



retain = [#"sex", "agecat",
          #"IRFAS",
          #"IRRELFAS_LMH",
          "IOTF4",
        "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9",         "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
        "teacheraccept", "teachercare", "teachertrust",
    "bulliedothers", "beenbullied",
    "cbulliedothers", "cbeenbullied",
    "fight12m", "injured12m",
    "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
    "emconlpref1", "emconlpref2", "emconlpref3",
    "friendhelp", "friendcounton", "friendshare", "friendtalk",
    "famhelp", "famsup", "famtalk", "famdec",
    "talkfather", "talkstepfa", "talkmother", "talkstepmo",
#    "timeexe",
    #    "health", "headache", "stomachache", "backache",
    #    "feellow", "irritable", "nervous","sleepdificulty", "dizzy","thinkbody",
    "lifesat",
#    "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2",
    "employfa", "employmo",
#    "employnotfa", "employnotmo"
]

retain_dict = {
    "demographics": [
        "sex", "agecat"
    ],
    "family_affluence_indices": [
        "IRFAS", #"IRRELFAS_LMH"
    ],
    "anthropometric_indices": [
        "IOTF4"
    ],
    "digital_communication": [
        "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9"
    ],
    "school_environment": [
        "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
        "teacheraccept", "teachercare", "teachertrust"
    ],
    "bullying_violence": [
        "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m"
    ],
    "digital_contact_frequency": [
        "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4"
    ],
    "digital_contact_preference": [
        "emconlpref1", "emconlpref2", "emconlpref3"
    ],
    "friend_support": [
        "friendhelp", "friendcounton", "friendshare", "friendtalk",
    ],
    "family_support": [
        "famhelp", "famsup", "famtalk", "famdec"
    ],
    "mom_support": [
        "talkmother", "talkstepmo"
    ],
    "father_support": [
        "talkfather", "talkstepfa"
    ],
    "health_behaviors": [
        "timeexe"
    ],
    "mental_health": [
        "lifesat", "feellow", "irritable", "nervous", "thinkbody"
    ],
    "physical_health": [
        "health", "headache", "stomachache", "backache","sleepdificulty", "dizzy"
    ],
    "family_structure": [
        "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1", "fosterhome1", "elsehome1_2"
    ],
    "family_employment": [
        "employfa", "employmo",
        #"employnotfa", "employnotmo"
    ]
}




if __name__ == "__main__":
    retain = [
        # "sex", "agecat", "IRFAS",
        # "IRRELFAS_LMH",
        #       "IOTF4",
        #       "emcsocmed1", "emcsocmed2", "emcsocmed3", "emcsocmed4", "emcsocmed5", "emcsocmed6", "emcsocmed7", "emcsocmed8", "emcsocmed9",
        #       "likeschool", "schoolpressure", "studtogether", "studhelpful", "studaccept",
        #     "teacheraccept", "teachercare", "teachertrust",
        #     "bulliedothers", "beenbullied", "cbulliedothers", "cbeenbullied", "fight12m", "injured12m",
        #     "emconlfreq1", "emconlfreq2", "emconlfreq3", "emconlfreq4",
        #     "emconlpref1", "emconlpref2", "emconlpref3",
        #     "friendhelp", "friendcounton", "friendshare", "friendtalk", "famhelp", "famsup", "famtalk", "famdec",
        #     "talkfather", "talkstepfa", "talkmother", "talkstepmo", "timeexe",        "health", "lifesat", "headache", "stomachache", "backache", "feellow", "irritable", "nervous",
        "sleepdificulty", "dizzy", "thinkbody", "motherhome1", "fatherhome1", "stepmohome1", "stepfahome1",
        "fosterhome1", "elsehome1_2",
        # "employfa", "employmo", "employnotfa", "employnotmo", "fasfamcar",    "fasbedroom",    "fascomputers",    "fasbathroom",    "fasdishwash",    "fasholidays"
    ]

    data = data_loader(retain)
