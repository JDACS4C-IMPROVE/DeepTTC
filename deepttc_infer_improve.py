import sys
from pathlib import Path
from typing import Dict

# [Req] IMPROVE/CANDLE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from model_params_def import infer_params
from DeepTTC_candle import *

# Model-specific imports, as needed
import os
import torch
# import pickle
import pandas as pd

filepath = Path(__file__).resolve().parent  # [Req]

def determine_device(cuda_name_from_params):
    """Determine device to run PyTorch functions.

    PyTorch functions can run on CPU or on GPU. In the latter case, it
    also takes into account the GPU devices requested for the run.

    :params str cuda_name_from_params: GPUs specified for the run.

    :return: Device available for running PyTorch functionality.
    :rtype: str
    """
    cuda_avail = torch.cuda.is_available()
    print("GPU available: ", cuda_avail)
    if cuda_avail:  # GPU available
        # -----------------------------
        # CUDA device from env var
        cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_env_visible is not None:
            # Note! When one or multiple device numbers are passed via
            # CUDA_VISIBLE_DEVICES, the values in python script are reindexed
            # and start from 0.
            print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
            cuda_name = "cuda:0"
        else:
            cuda_name = cuda_name_from_params
        device = cuda_name
    else:
        device = "cpu"

    return device

# [Req]
def run(params:Dict):
    """ Run model inference.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for test set and build model path
    # --------------------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(params['data_format'], stage="test")
    
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                    model_file_format=params["model_file_format"],
                                    model_dir=params["input_model_dir"])
    
    # --------------------------------------------------------------------
    # Load inference data (ML data)
    # --------------------------------------------------------------------
    test_data = {}
    test_data['drug'] = pd.read_hdf(os.path.join(params["input_data_dir"],test_data_fname), key='drug')
    test_data['gene_expression'] = pd.read_hdf(os.path.join(params["input_data_dir"],
        test_data_fname), key='gene_expression')
    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    # device = determine_device(params["cuda_name"])
    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------

    model = DeepTTC(modeldir=modelpath, args=params)
    model.load_pretrained(modelpath)
    # Compute predictions
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = model.predict(
        test_data['drug'], test_data['gene_expression'])

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_label,
        y_pred=y_pred,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=y_label,
            y_pred=y_pred,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepttc_params.txt",
        additional_definitions=infer_params
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
