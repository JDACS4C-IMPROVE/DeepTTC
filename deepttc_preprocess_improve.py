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
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
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

    # --------------------------------------------------------------------
    # [Req] Create dataloaders
    # --------------------------------------------------------------------
    omics_loader = omics_utils.OmicsLoader(params)
    drugs_loader = drugs_utils.DrugsLoader(params)

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------

    df_cell_all = omics_loader.dfs['cancer_gene_expression.tsv']
    df_drug_all = drugs_loader.dfs['drug_SMILES.tsv']
    df_drug_all = df_drug_all.reset_index()
    df_drug_all.columns = [params["drug_col_name"], "SMILES"]
    params['drug_id'] = params["drug_col_name"]

    if params["use_lincs"]:
        genes_fpath = filepath/"landmark_genes"
        df_cell_all = gene_selection(df_cell_all, genes_fpath, canc_col_name=params["canc_col_name"])

    stages = {"train": params["train_split_file"],
            "val": params["val_split_file"],
            "test": params["test_split_file"]}
    scaler = None
    for stage, split_file in stages.items():
        print(f"Building stage: {stage}")
        df_response = drp.DrugResponseLoader(params,
                                            split_file=stages[stage],
                                            verbose=False).dfs["response.tsv"]

        df_y, df_cell = get_common_samples(df1=df_response,
                                        df2=df_cell_all,
                                        ref_col=params["canc_col_name"])
        print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

        # Normalize features using training set
        if stage == "train":  # Ignore scaler object even if specified
            df_cell, scaler = scale_df(df_cell, scaler_name=params["scaling"])
            if params["scaling"] is not None and params["scaling"] != "none":
                # Store normalization object
                scaler_fname = os.path.join(params["output_dir"], "cell_xdata_scaler.gz")
                joblib.dump(scaler, scaler_fname)
                print("Scaling object created is stored in: ", scaler_fname)
        else:
            # Use passed scikit scaler object
            df_cell, _ = scale_df(df_cell, scaler=scaler)

        # Sub-select desired response column (y_col_name)
        # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
        df_y = df_y[[params["drug_col_name"], params["canc_col_name"], params["y_col_name"]]]
        df_y['Label'] = df_y[params['y_col_name']]
        # Combine data

        obj = DataEncoding(params, params["input_supp_data_dir"], params["canc_col_name"],
                            params["sample_col_name"], params["y_col_name"], params["drug_col_name"])

        df_drug_stage = df_drug_all[df_drug_all[params['drug_col_name']].isin(df_y[params['drug_col_name']])]

        smile_encode = pd.Series(df_drug_all['SMILES'].unique()).apply(obj._drug2emb_encoder)
        uniq_smile_dict = dict(zip(df_drug_all['SMILES'].unique(), smile_encode))

        df_drug_stage['drug_encoding'] = [uniq_smile_dict[i] for i in df_drug_stage['SMILES']]
        #df_drug_stage = df_drug_stage.reset_index()

        df_drug_stage = pd.merge(df_y, df_drug_stage, on=params["drug_col_name"], how='inner')

        #df_drug_stage.index = range(df_drug_stage.shape[0])
        #df_cell.index = range(df_cell.shape[0])

        #df_drug_stage = df_drug_stage.drop(['index'], axis=1)
        drug_columns = ['drug_encoding']
        data = pd.merge(df_cell, df_drug_stage, on=params["canc_col_name"], how='inner')
        df_cell = df_cell.drop([params["canc_col_name"]], axis=1)
        gene_expression_columns = df_cell.columns

        # --------------------------------------------------------------------
        # [MODEL] Save X data
        # --------------------------------------------------------------------
        # Save the subset of y data
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
