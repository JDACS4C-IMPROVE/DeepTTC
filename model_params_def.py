from improvelib.utils import str2bool

preprocess_params = [
    {"name": "sample_col_name",
     "type": str,
     "default": "COSMIC_ID",
     "help": "ID format of the samples",
     },
    {
    "name": "gene_dtype",
    "type": str,
    "default": None,
    "help": "Specify the floating point precision for gene expression data (float32, float16, or None for default float64).",
    },
]

train_params = [
    {
        "name": "save_data",
        "type": str2bool,
        "default": False,
        "help": "Whether to save loaded data in pickle files",
    },
    {
        "name": "use_lincs",
        "type": str2bool,
        "default": False,
        "help": "Whether to use a LINCS subset of genes ONLY",
    },
    {
        "name": "benchmark_dir",
        "type": str,
        "default": None,
        "help": "Directory with the input data for benchmarking",
    },
    {
        "name": "benchmark_result_dir",
        "type": str,
        "default": None,
        "help": "Directory for benchmark output",
    },
    {
        "name": "generate_input_data",
        "type": str2bool,
        "default": None,
        "help": "'True' for generating input data anew, 'False' for using stored data",
    },
    {
        "name": "mode",
        "type": str,
        "default": None,
        "help": "Execution mode. Available modes are: 'run', 'benchmark'",
    },
    {
        "name": "cancer_id",
        "type": str,
        "default": None,
        "help": "Column name for cancer",
    },
    {
        "name": "drug_id",
        "type": str,
        "default": None,
        "help": "Column name for drug",
    },
    {
        "name": "sample_id",
        "type": str,
        "default": None,
        "help": "Column name for samples/cell lines",
    },
    {
        "name": "target_id",
        "type": str,
        "default": None,
        "help": "Column name for target",
    },
    {
        "name": "train_data_drug",
        "type": str,
        "default": None,
        "help": "Drug data for training",
    },
    {
        "name": "test_data_drug",
        "type": str,
        "default": None,
        "help": "Drug data for testing",
    },
    {
        "name": "train_data_rna",
        "type": str,
        "default": None,
        "help": "RNA data for training",
    },
    {
        "name": "test_data_rna",
        "type": str,
        "default": None,
        "help": "RNA data for testing",
    },
    {
        "name": "vocab_dir",
        "type": str,
        "default": None,
        "help": "Directory with ESPF vocabulary",
    },
    {
        "name": "transformer_num_attention_heads_drug",
        "type": int,
        "default": None,
        "help": "number of attention heads for drug transformer",
    },
    {
        "name": "input_dim_drug",
        "type": int,
        "default": None,
        "help": "Input size of the drug transformer",
    },
    {
        "name": "transformer_emb_size_drug",
        "type": int,
        "default": None,
        "help": "Size of the drug embeddings",
    },
    {
        "name": "transformer_n_layer_drug",
        "type": int,
        "default": None,
        "help": "Number of layers for drug transformer",
    },
    {
        "name": "transformer_intermediate_size_drug",
        "type": int,
        "default": None,
        "help": "Intermediate size of the drug layers",
    },
    {
        "name": "transformer_attention_probs_dropout",
        "type": float,
        "default": None,
        "help": "number of layers for drug transformer",
    },
    {
        "name": "transformer_hidden_dropout_rate",
        "type": float,
        "default": None,
        "help": "dropout rate for transformer hidden layers",
    },
    {
        "name": "dropout",
        "type": float,
        "default": None,
        "help": "dropout rate for common part",
    },
    {
        "name": "input_dim_drug_classifier",
        "type": int,
        "default": None,
        "help": "input dimensions for drug classifier",
    },
    {
        "name": "input_dim_gene_classifier",
        "type": int,
        "default": None,
        "help": "input dimensions for gene classifier",
    },
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Specify the CUDA device to use for model training and inference.",
    },
]

infer_params = [
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Specify the CUDA device to use for model training and inference.",
    },  
    {
        "name": "input_dim_drug",
        "type": int,
        "default": None,
        "help": "Input size of the drug transformer",
    },
    {
        "name": "transformer_emb_size_drug",
        "type": int,
        "default": None,
        "help": "Size of the drug embeddings",
    },
    {
        "name": "transformer_n_layer_drug",
        "type": int,
        "default": None,
        "help": "Number of layers for drug transformer",
    },
    {
        "name": "transformer_intermediate_size_drug",
        "type": int,
        "default": None,
        "help": "Intermediate size of the drug layers",
    },
    {
        "name": "transformer_attention_probs_dropout",
        "type": float,
        "default": None,
        "help": "number of layers for drug transformer",
    },
    {
        "name": "transformer_hidden_dropout_rate",
        "type": float,
        "default": None,
        "help": "dropout rate for transformer hidden layers",
    },
        {
        "name": "dropout",
        "type": float,
        "default": None,
        "help": "dropout rate for common part",
    },
    {
        "name": "transformer_num_attention_heads_drug",
        "type": int,
        "default": None,
        "help": "number of attention heads for drug transformer",
    },
    {
        "name": "input_dim_drug_classifier",
        "type": int,
        "default": None,
        "help": "input dimensions for drug classifier",
    },
    {
        "name": "input_dim_gene_classifier",
        "type": int,
        "default": None,
        "help": "input dimensions for gene classifier",
    },
]
