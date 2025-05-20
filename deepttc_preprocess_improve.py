import sys
from pathlib import Path
from typing import Dict

# [MODEL] Model-specific imports, as needed
import subprocess
import joblib
import pandas as pd
import numpy as np
# import pickle
import os
from Step2_DataEncoding import DataEncoding
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# [Req] Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
# [Req] Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drp_utils as drp
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent  # [Req]


def gene_selection(df, genes_fpath, canc_col_name):
    """ Takes a dataframe omics data (e.g., gene expression) and retains only
    the genes specified in genes_fpath.
    """
    with open(genes_fpath) as f:
        genes = [str(line.rstrip()) for line in f]
    genes = sorted(list(set(genes).intersection(set(df.columns[1:]))))
    cols = [canc_col_name] + genes
    return df[cols]

def get_common_samples(df1, df2, ref_col):
    # Retain df1 and df2 samples with common ref_col
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def scale_df(dataf, scaler_name="std", scaler=None, verbose=False):
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return dataf, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = dataf.select_dtypes(include="number")

    if scaler is None:  # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(
                f"The specified scaler {scaler_name} is not implemented (no df scaling).")
            return dataf, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else:  # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    dataf[df_num.columns] = df_norm
    return dataf, scaler




def run(params:Dict):
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    # not needed for this data/model

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    print("Load omics data.")
    ge = drp.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])
    print("Load drug data.")
    smiles = drp.get_x_data(file = params['drug_smiles_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    #smiles.columns = ["SMILES"]
    #smiles = smiles.reset_index()

    print("Load train response data.")
    response_train = drp.get_response_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   response_file=params['y_data_file'])
    print("Find intersection of training data.")
    response_train = drp.get_response_with_features(response_train, ge, params['canc_col_name'])
    response_train = drp.get_response_with_features(response_train, smiles, params['drug_col_name'])
    ge_train = drp.get_features_in_response(ge, response_train, params['canc_col_name'])

    print("Determine transformations.")
    drp.determine_transform(ge_train, 'ge_transform', params['cell_transcriptomic_transform'], params['output_dir'])

    # ------------------------------------------------------
    # [Req] Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------
    # Dict with split files corresponding to the three sets (train, val, and test)

    stages = {"train": params["train_split_file"],
            "val": params["val_split_file"],
            "test": params["test_split_file"]}
    
    for stage, split_file in stages.items():
        print(f"Prepare data for stage {stage}.")
        print(f"Find intersection of {stage} data.")
        response_stage = drp.get_response_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                response_file=params['y_data_file'])
        response_stage = drp.get_response_with_features(response_stage, ge, params['canc_col_name'])
        response_stage = drp.get_response_with_features(response_stage, smiles, params['drug_col_name'])
        ge_stage = drp.get_features_in_response(ge, response_stage, params['canc_col_name'])
        smiles_stage = drp.get_features_in_response(smiles, response_stage, params['drug_col_name'])

        print(f"Transform {stage} data.")
        ge_stage = drp.transform_data(ge_stage, 'ge_transform', params['output_dir'])

        # Preprocess drug data
        obj = DataEncoding(params, params["input_supp_data_dir"], params["canc_col_name"],
                            params["sample_col_name"], params["y_col_name"], params["drug_col_name"])
        smile_encode = pd.Series(smiles_stage['SMILES'].unique()).apply(obj._drug2emb_encoder)
        uniq_smile_dict = dict(zip(smiles_stage['SMILES'].unique(), smile_encode))
        smiles_stage['drug_encoding'] = [uniq_smile_dict[i] for i in smiles_stage['SMILES']]

        print(f"Merge {stage} data")
        data = pd.merge(response_stage, smiles_stage, on=params["drug_col_name"], how='inner')
        data = pd.merge(ge_stage, data, on=params["canc_col_name"], how='inner')
        ge_stage = ge_stage.drop([params["canc_col_name"]], axis=1) # should be index
        gene_expression_columns = ge_stage.columns
        drug_columns = ['drug_encoding']

        # --------------------------------------------------------------------
        # [MODEL] Save X data
        # --------------------------------------------------------------------
        df_gene_expression = data[gene_expression_columns]

        # Convert dtype if provided
        if params["gene_dtype"] is not None:
            if params["gene_dtype"] not in ["float32", "float16"]:
                raise ValueError("dtype must be 'float32' or 'float16'")
            df_gene_expression = df_gene_expression.astype(params["gene_dtype"])

        df_drug = data[drug_columns]
        df_label = data[params['y_col_name']]

        out_path = os.path.join(params["output_dir"], frm.build_ml_data_file_name(params["data_format"], stage=stage))
        df_output = {'drug': df_drug, 'gene_expression': df_gene_expression, 'label': df_label}
        for key in df_output:
            df_output[key].to_hdf(out_path, key)

        # --------------------------------------------------------------------
        # [Req] Save response data (Y data)
        # --------------------------------------------------------------------
        y_df = pd.DataFrame(data[[params['y_col_name'], params['canc_col_name'], params['drug_col_name']]])
        frm.save_stage_ydf(y_df, stage, params['output_dir'])

    return params["output_dir"]


def main(args):
    # [Req]
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        filepath,
        default_config="deepttc_params.txt",
        additional_definitions=preprocess_params)

    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
