import sys
from functools import partial

from superglue_modules.xlnet_module import (
    XLNetContactLastCLSWithTwoTokensModule,
    XLNetModule,
)
from superglue_modules.wsc_module import SpanClassifierModule
from task_config import SuperGLUE_LABEL_MAPPING, SuperGLUE_TASK_METRIC_MAPPING
from torch import nn

from snorkel.mtl.scorer import Scorer
from snorkel.mtl.task import Task, Operation

from . import utils

sys.path.append("..")  # Adds higher directory to python modules path.


TASK_NAME = "WSC"


def build_task(xlnet_model_name, last_hidden_dropout_prob=None):
    if last_hidden_dropout_prob:
        raise NotImplementedError(f"TODO: last_hidden_dropout_prob for {TASK_NAME}")

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
                f"{TASK_NAME}_pred_head": SpanClassifierModule(
                    d_inp=xlnet_output_dim, proj_dim=xlnet_output_dim // 2
                ),
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
                name=f"{TASK_NAME}_pred_head",
                module_name=f"{TASK_NAME}_pred_head",
                inputs=[
                    (f"{TASK_NAME}_xlnet_module", 0),
                    ("_input_", "token1_idx"),
                    ("_input_", "token2_idx"),
                    ("_input_", "token_masks"),
                ],
            ),
        ],
        loss_func=loss_fn,
        output_func=output_fn,
        scorer=Scorer(metrics=metrics, custom_metric_funcs=custom_metric_funcs),
    )

    return task
