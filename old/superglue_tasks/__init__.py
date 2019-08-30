from . import cb, copa, multirc, rte, wic, wsc, swag, mrpc, sst

task_funcs = {
    "CB": cb.build_task,
    "COPA": copa.build_task,
    "MultiRC": multirc.build_task,
    "RTE": rte.build_task,
    "WiC": wic.build_task,
    "WSC": wsc.build_task,
    "SWAG": swag.build_task,
    "MRPC": mrpc.build_task,
    "SST": sst.build_task,
}
