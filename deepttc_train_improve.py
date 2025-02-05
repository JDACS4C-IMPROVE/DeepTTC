import sys
from pathlib import Path
from typing import Dict

# Model-specific imports
import os
import json
# import pickle
import pandas as pd
from DeepTTC_candle import get_model

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics
from model_params_def import train_params

filepath = Path(__file__).resolve().parent # [Req]

def compute_performace_scores(y_true, y_pred, metrics, outdtd, stage):
    """Evaluate predictions according to specified metrics.

    Metrics are evaluated. Scores are stored in specified path and returned.

    :params array y_true: Array with ground truth values.
    :params array y_pred: Array with model predictions.
    :params listr metrics: List of strings with metrics to evaluate.
    :params Dict outdtd: Dictionary with path to store scores.
    :params str stage: String specified if evaluation is with respect to
            validation or testing set.

    :return: Python dictionary with metrics evaluated and corresponding scores.
    :rtype: dict
    """
    scores = compute_metrics(y_true, y_pred, metrics)
    key = f"{stage}_loss"
    scores[key] = scores["mse"]

    with open(outdtd["scores"], "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    if stage == "val":
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
        print("Validation scores:\n\t{}".format(scores))
    elif stage == "test":
        print("Inference scores:\n\t{}".format(scores))
    return scores


def run(params:Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]
    
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                     model_file_format=params["model_file_format"],
                                     model_dir=params["output_dir"])
    
    # --------------------------------------------------------------------
    # Load model input data (ML data) for train and val
    # --------------------------------------------------------------------
    train_data = {}
    train_data['drug'] = pd.read_hdf(os.path.join(params["input_dir"],train_data_fname), key='drug')
    train_data['gene_expression'] = pd.read_hdf(os.path.join(params["input_dir"],train_data_fname), key='gene_expression')
    val_data = {}
    val_data['drug'] = pd.read_hdf(os.path.join(params["input_dir"],val_data_fname), key='drug')
    val_data['gene_expression'] = pd.read_hdf(os.path.join(params["input_dir"],val_data_fname), key='gene_expression')

    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    
    # --------------------------------------------------------------------
    # Prepare model
    # --------------------------------------------------------------------
    model = get_model(params)
    
    # --------------------------------------------------------------------
    # Train. Iterate over epochs.
    # --------------------------------------------------------------------
    model = model.train(train_drug=train_data['drug'], train_rna=train_data['gene_expression'],
                        val_drug=val_data['drug'], val_rna=val_data['gene_expression'])
    print(f'Saving model to {modelpath}')
    model.save_model(modelpath)
    print("Model Saved :{}".format(modelpath))
    
    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------
    model.load_pretrained(modelpath)
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = model.predict(
        val_data['drug'], val_data['gene_expression'])

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_label,
        y_pred=y_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=y_label,
        y_pred=y_pred,
        stage="val",
        output_dir=params["output_dir"],
        metric_type=params["metric_type"]
    )

    return val_scores


def main(args):
    filepath = Path(__file__).resolve().parent

    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="deepttc_params.txt",
                                       additional_definitions=train_params
                                       )
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
