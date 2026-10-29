from parser.parser import prepare_all_data


prepare_all_data(ct_dir="datasets/Center2/CT images/Patient01",
                 counters_filename="datasets/Center2/Treatment plans/Patient01/RS.dcm",
                 dose_filename="datasets/Center2/Treatment plans/Patient01/RD.dcm",
                 plan_filename="datasets/Center2/Treatment plans/Patient01/RP.dcm")

#TODO: Сверка файлов по Serius UID
#TODO: Объединение данных скан+контур с дозовой картой
#TODO: Единая функция обработчика путей в checker