#!/usr/bin/env bash
sample=$1
level=$2
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/hep_xml2json.py ${sample}.xml ${sample}.json

###### obtain TME region mask and no-TME region mask
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/ROI_mask.py ${sample}.svs ${sample}.json ${sample}.npy

####obtain TME region mask
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/Get_grid_points.py ${sample}.svs ./ --level ${level} --mask_path ${sample}.Tumor.mask.npy

#### obtain coordinate points of no-TME region mask
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/Get_grid_points.py ${sample}.svs ./ --level ${level} --mask_path ${sample}.No_Tumor.mask.npy
