from . import cb, copa, multirc, rte, wic, wsc, swag, mrpc, sst

parser = {
    "MultiRC": multirc.parse,
    "WiC": wic.parse,
    "CB": cb.parse,
    "COPA": copa.parse,
    "RTE": rte.parse,
    "WSC": wsc.parse,
    "SWAG": swag.parse,
    "MRPC": mrpc.parse,
    "SST": sst.parse,
}
