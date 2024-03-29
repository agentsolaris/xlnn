import sys
from functools import partial

from torch import nn

from snorkel.model.metrics import metric_score
from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Operation, Task
from superglue_modules.xlnet_module import XLNetLastCLSModule, XLNetModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "MultiRC"


def build_task(xlnet_model_name, last_hidden_dropout_prob=0.0):

    xlnet_module = XLNetModule(xlnet_model_name)
    xlnet_output_dim = 768 if "base" in xlnet_model_name else 1024

    task_cardinality = (
        len(SuperGLUE_LABEL_MAPPING[TASK_NAME].keys())
        if SuperGLUE_LABEL_MAPPING[TASK_NAME] is not None
        else 1
    )

    metrics = (
        SuperGLUE_TASK_METRIC_MAPPING[TASK_NAME]
        if TASK_NAME in SuperGLUE_TASK_METRIC_MAPPING
        else []
    )

    custom_metric_funcs = {}

    loss_fn = partial(utils.ce_loss, f"{TASK_NAME}_pred_head")
    output_fn = partial(utils.output, f"{TASK_NAME}_pred_head")

    task = Task(
        name=TASK_NAME,
        module_pool=nn.ModuleDict(
            {
                "xlnet_module": xlnet_module,
                "xlnet_last_CLS": XLNetLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                ),
                f"{TASK_NAME}_pred_head": nn.Linear(xlnet_output_dim, task_cardinality),
            }
        ),
        task_flow=[
            Operation(
                name=f"{TASK_NAME}_xlnet_module",
                module_name="xlnet_module",
                inputs=[
                    ("_input_", "token_ids"),
                    ("_input_", "token_segments"),
                    ("_input_", "token_masks"),
                ],
            ),
            Operation(
                name=f"{TASK_NAME}_xlnet_last_CLS",
                module_name="xlnet_last_CLS",
                inputs=[(f"{TASK_NAME}_xlnet_module", 0)],
            ),
            Operation(
                name=f"{TASK_NAME}_pred_head",
                module_name=f"{TASK_NAME}_pred_head",
                inputs=[(f"{TASK_NAME}_xlnet_last_CLS", 0)],
            ),
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, custom_metric_funcs=custom_metric_funcs),
    )

    return task
