#!/usr/bin/env python
# coding: utf-8

# hyperparameter optimization with raytune for disease classification

# imports
import os
import subprocess

GPU_NUMBER = [0, 1, 2, 3, 4]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"
# os.environ["LD_LIBRARY_PATH"] = "/path/to/miniconda3/lib:/path/to/sw/lib:/path/to/sw/lib"

# initiate runtime environment for raytune
import ray
from ray import tune

# from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.search.hyperopt import HyperOptSearch

ray.shutdown()  # engage new ray session

# runtime_env = {"conda": "base",
#               "env_vars": {"LD_LIBRARY_PATH": "/path/to/miniconda3/lib:/path/to/sw/lib:/path/to/sw/lib"}}
runtime_env = {"pip": "/path/to/saved/requeirement/file/requirements.txt"}

ray.init(runtime_env=runtime_env)


def initialize_ray_with_check(ip_address):
    """
    Initialize Ray with a specified IP address and check its status and accessibility.

    Args:
    - ip_address (str): The IP address (with port) to initialize Ray.

    Returns:
    - bool: True if initialization was successful and dashboard is accessible, False otherwise.
    """
    try:
        ray.init(address=ip_address)
        print(ray.nodes())

        services = ray.get_webui_url()
        if not services:
            raise RuntimeError("Ray dashboard is not accessible.")
        else:
            print(f"Ray dashboard is accessible at: {services}")
        return True
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        return False


# Usage:
ip = "192.168.xxx.yyy:zzzz"  # Replace with your actual IP address and port
if initialize_ray_with_check(ip):
    print("Ray initialized successfully.")
else:
    print("Error during Ray initialization.")

import datetime

import numpy as np
import seaborn as sns

sns.set()
from collections import Counter

from geneformer import DataCollatorForCellClassification
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments

from datasets import load_from_disk

# number of CPU cores
num_proc = 30

# load train dataset with columns:
# cell_type (annotation of each cell's type)
# disease (healthy or disease state)
# individual (unique ID for each patient)
# length (length of that cell's rank value encoding)

dataset_name = "/path/to/seved/dataset/xxx.dataset/"

train_dataset = load_from_disk(dataset_name)

# check column names
try:
    print(np.unique(train_dataset["cell_type"]))
except KeyError as e:
    print("KeyError: {}".format(e))
    print("changing to cell_type")
    train_dataset = train_dataset.rename_column(
        "column name in cell types infomation", "cell_type"
    )
    print("change finished")
    print(np.unique(train_dataset["cell_type"]))

try:
    print(np.unique(train_dataset["disease"]))
except KeyError as e:
    print("KeyError: {}".format(e))
    print("changing to disease")
    train_dataset = train_dataset.rename_column(
        "column name in diseases infomation", "disease"
    )
    print("change finished")
    print(np.unique(train_dataset["disease"]))


# filter dataset for given cell_type
def if_cell_type(example):
    return example["cell_type"].startswith("nan")


trainset_v2 = train_dataset.filter(if_cell_type, num_proc=num_proc)

# create dictionary of disease states : label ids


target_names = ["Cande", "KO", "WT"]

target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))

trainset_v2_shuffled = trainset_v2.shuffle(seed=42)
trainset_v3 = trainset_v2_shuffled.rename_column("disease", "label")


# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example


trainset_v4 = trainset_v3.map(classes_to_ids, num_proc=num_proc)

trainset = trainset_v4.select([i for i in range(0, round(len(trainset_v4) * 0.8))])
validset = trainset_v4.select(
    [i for i in range(round(len(trainset_v4) * 0.8), len(trainset_v4))]
)

print("trainset: {}".format(trainset))
print("validset: {}".format(validset))

train_indiv = list(Counter(trainset["label"]).keys())
valid_indiv = list(Counter(validset["label"]).keys())


def if_train(example):
    return example["label"] in train_indiv


classifier_trainset = trainset.filter(if_train, num_proc=num_proc).shuffle(seed=42)


def if_valid(example):
    return example["label"] in valid_indiv


classifier_validset = validset.filter(if_valid, num_proc=num_proc).shuffle(seed=42)

print("classifier_trainset: {}".format(classifier_trainset))
print(
    "classifier_trainset['label']: {}".format(np.unique(classifier_trainset["label"]))
)
print("classifier_validset: {}".format(classifier_validset))
print(
    "classifier_validset['label']: {}".format(np.unique(classifier_validset["label"]))
)

# define output directory path
current_date = datetime.datetime.now()
datestamp = (
    f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
)

output_dir = f"/path/to/output/directory/{datestamp}_mouse-geneformer_DiseaseClassifie/"

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, "pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f"mkdir {output_dir}", shell=True)

# set training parameters
# how many pretrained layers to freeze
freeze_layers = 3
# batch size for training and eval
geneformer_batch_size = 12
# number of epochs
epochs = 2
# logging steps
logging_steps = 100


# define function to initiate model
def model_init():
    pretrain_model = "your pretrained model"

    model = BertForSequenceClassification.from_pretrained(
        "/path/to/pretrained/model/{}/models/".format(pretrain_model),
        num_labels=len(target_names),
        output_attentions=False,
        output_hidden_states=False,
    )
    if freeze_layers is not None:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    model = model.to("cuda")
    return model


# define metrics
# note: macro f1 score recommended for imbalanced multiclass classifiers
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


# set training arguments
training_args = {
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "steps",
    "eval_steps": logging_steps,
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": True,
    "skip_memory_metrics": True,  # memory tracker causes errors in raytune
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
}

training_args_init = TrainingArguments(**training_args)

# create the trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=classifier_trainset,
    eval_dataset=classifier_validset,
    compute_metrics=compute_metrics,
)

# specify raytune hyperparameter search space
ray_config = {
    "num_train_epochs": tune.choice([epochs]),
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "weight_decay": tune.uniform(0.0, 0.3),
    "lr_scheduler_type": tune.choice(["linear", "cosine", "polynomial"]),
    "warmup_steps": tune.uniform(100, 10000),
    "seed": tune.uniform(0, 100),
    "per_device_train_batch_size": tune.choice([geneformer_batch_size]),
}

hyperopt_search = HyperOptSearch(metric="eval_accuracy", mode="max")

# optimize hyperparameters
trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    resources_per_trial={"cpu": 8, "gpu": 1},
    hp_space=lambda _: ray_config,
    search_alg=hyperopt_search,
    n_trials=100,  # number of trials
    progress_reporter=tune.CLIReporter(
        max_report_frequency=600,
        sort_by_metric=True,
        max_progress_rows=100,
        mode="max",
        metric="eval_accuracy",
        metric_columns=["loss", "eval_loss", "eval_accuracy"],
    ),
)
