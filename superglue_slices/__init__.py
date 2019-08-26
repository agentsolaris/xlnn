from . import general_sfs, RTE_sfs, WiC_sfs, CB_sfs, COPA_sfs, MultiRC_sfs, WSC_sfs, MRPC_sfs, SSS_sfs

slice_func_dict = {
    "CB": CB_sfs.slice_func_dict,
    "COPA": COPA_sfs.slice_func_dict,
    "MultiRC": MultiRC_sfs.slice_func_dict,
    "RTE": RTE_sfs.slice_func_dict,
    "WiC": WiC_sfs.slice_func_dict,
    "WSC": WSC_sfs.slice_func_dict,
    "MRPC": MRPC_sfs.slice_func_dict,
    "SST": SST_sfs.slice_func_dict,
}
