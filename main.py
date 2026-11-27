import pprint
from parser.parser import *


for id in range(1, 21):
    cts, masks, dose, valid_masks, info = prepare_all_data(ct_dir=f"datasets/Center2/CT images/Patient{id:02}",
                                                           counters_filename=f"datasets/Center2/Treatment plans/Patient{id:02}/RS.dcm",
                                                           dose_filename=f"datasets/Center2/Treatment plans/Patient{id:02}/RD.dcm",
                                                           plan_filename=f"datasets/Center2/Treatment plans/Patient{id:02}/RP.dcm")
    save_ct_and_masks(cts, masks, dose, valid_masks,
                      f"output/Patient{id:02}/data.npz")
    save_info_json(info, f"output/Patient{id:02}/info.json")
    print(f"End Patient{id:02}")
