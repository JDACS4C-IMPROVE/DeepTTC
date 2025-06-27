import sys
from pathlib import Path
from typing import Dict

# [MODEL] Model-specific imports, as needed
import pandas as pd
import os
from Step2_DataEncoding import drug2emb_encoder

# [Req] Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
import improvelib.utils as frm
# [Req] Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drp_utils as drp
from model_params_def import preprocess_params

filepath = Path(__file__).resolve().parent  # [Req]


def run(params:Dict):
    print("Load omics data.")
    ge = frm.get_x_data(file = params['cell_transcriptomic_file'], 
                                        benchmark_dir = params['input_dir'], 
                                        column_name = params['canc_col_name'])
    print("Load drug data.")
    smiles = frm.get_x_data(file = params['drug_smiles_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
    smiles.columns = ['SMILES']
    # ------------------------------------------------------
    # [Req] Validity check of feature representations
    # ------------------------------------------------------
    smi_to_drop = []
    for i, row in smiles.iterrows():
        try:
            smi = drug2emb_encoder(row['SMILES'])
        except:
            print(f"Invalid SMILE string {row['SMILES']}, ID is {i}, removing from analysis.")
            smi_to_drop = smi_to_drop + [i]
            
    smiles = smiles.drop(smi_to_drop)

    # ------------------------------------------------------
    # [Req] Determine preprocessing on training data
    # ------------------------------------------------------
    print("Load train response data.")
    response_train = frm.get_y_data(split_file=params["train_split_file"], 
                                   benchmark_dir=params['input_dir'], 
                                   y_data_file=params['y_data_file'])
    print("Find intersection of training data.")
    response_train = frm.get_y_data_with_features(response_train, ge, params['canc_col_name'])
    response_train = frm.get_y_data_with_features(response_train, smiles, params['drug_col_name'])
    ge_train = frm.get_features_in_y_data(ge, response_train, params['canc_col_name'])

    print("Determine transformations.")
    frm.determine_transform(ge_train, 'ge_transform', params['cell_transcriptomic_transform'], params['output_dir'])

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
        response_stage = frm.get_y_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                y_data_file=params['y_data_file'])
        response_stage = frm.get_y_data_with_features(response_stage, ge, params['canc_col_name'])
        response_stage = frm.get_y_data_with_features(response_stage, smiles, params['drug_col_name'])
        ge_stage = frm.get_features_in_y_data(ge, response_stage, params['canc_col_name'])
        smiles_stage = frm.get_features_in_y_data(smiles, response_stage, params['drug_col_name'])

        print(f"Transform {stage} data.")
        ge_stage = frm.transform_data(ge_stage, 'ge_transform', params['output_dir'])

        # Preprocess drug data
        #obj = DataEncoding(params, params["input_supp_data_dir"], params["canc_col_name"],
        #                    params["sample_col_name"], params["y_col_name"], params["drug_col_name"])
        smile_encode = pd.Series(smiles_stage['SMILES'].unique()).apply(drug2emb_encoder)
        uniq_smile_dict = dict(zip(smiles_stage['SMILES'].unique(), smile_encode))
        smiles_stage['drug_encoding'] = [uniq_smile_dict[i] for i in smiles_stage['SMILES']]

        print(f"Merge {stage} data")
        data = pd.merge(response_stage, smiles_stage, on=params["drug_col_name"], how='inner')
        data = pd.merge(ge_stage, data, on=params["canc_col_name"], how='inner')
        gene_expression_columns = ge_stage.columns
        print(gene_expression_columns)
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

# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="deepttc_params.ini",
                                       additional_definitions=preprocess_params)
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
