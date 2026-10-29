from parser.parser import prepare_all_data


prepare_all_data(ct_dir="datasets/Center2/CT images/Patient01",
                 counters_filename="datasets/Center2/Treatment plans/Patient01/RS.dcm",
                 dose_filename="datasets/Center2/Treatment plans/Patient01/RD.dcm",
                 plan_filename="datasets/Center2/Treatment plans/Patient01/RP.dcm")
