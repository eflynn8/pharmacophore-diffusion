import pickle
import rdkit
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()

## Specify path to input file and desired output pkl file path
parser.add_argument("-i", "--input_data", dest = "input_path", help = "Path to input file", required=True)
parser.add_argument("-o", "--output_pkl", dest = "output_path", help = "Path for output pkl file", required=False)

args = parser.parse_args()

## Fn for one-hot encoding pharmacophore features
## Aromatic, HydrogenDonor, HydrogenAcceptor, PositiveIon, NegativeIon, Hydrophobic
def one_hot_encode_pharms(arr):
    one_hot = np.zeros((len(arr), 6))
    one_hot[np.arange(len(arr)), arr] = 1
    return one_hot

## Fn for one-hot encoding protein features
## ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'D']
def one_hot_encode_prots(arr, enc):
    encoded_data = enc.fit_transform(arr.reshape(-1, 1)).toarray()
    return encoded_data

with open(args.input_path, 'rb') as datafile:
    data = pickle.load(datafile)

## Collect data entries into lists
rec_file_name = [prot[0] for prot in data]
lig_file_name = [ph[1] for ph in data]
lig_obj = [ph[2] for ph in data]
lig_pos = [lig[3][0] for lig in data]
lig_feat = [lig[3][1] for lig in data]
rec_pos = [rec[4][0] for rec in data]
rec_feat = [rec[4][1] for rec in data]

## Set up dictionary of relevant data + convert to DF
data_dict = {}
data_dict['prot_file_name'] = rec_file_name
data_dict['pharm_file_name'] = lig_file_name
data_dict['pharm_obj'] = lig_obj
data_dict['pharm_pos'] = lig_pos
data_dict['pharm_feat'] = lig_feat
data_dict['prot_pos'] = rec_pos
data_dict['prot_feat'] = rec_feat

df = pd.DataFrame.from_dict(data_dict)

print("Completed dataframe; starting post-processing.")

## One-hot encode pharma features and prot elements
df['pharm_feat'] = df['pharm_feat'].apply(one_hot_encode_pharms)

print("Finished one-hot encoding of pharmacophores; starting protein encoding.")

enc = OneHotEncoder(categories = [['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'D']])
df['prot_feat'] = df['prot_feat'].apply(one_hot_encode_prots, enc=enc)


if not args.output_path:
    output_path = args.input_path[:-4] + "_processed.pkl"

df.to_pickle(output_path)

