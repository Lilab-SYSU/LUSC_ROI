We will train a automatic extraction model of ROI for LUSC whole slide images.
1. training stage
we fistly perform preprocess of whole slide images.
Pathologist uses ASAP software to annotate ROI regions and export the coordinates of ROI regions into xml format files.
#####Convert xml format file to json format.
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/hep_xml2json.py ${sample}.xml ${sample}.json

###### Obtain ROI region mask and no-ROI region mask.
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/ROI_mask.py ${sample}.svs ${sample}.json ${sample}.npy

####Generat coordinate points in  ROI region.
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/Get_grid_points.py ${sample}.svs ./ --level ${level} --mask_path ${sample}.Tumor.mask.npy

#### Obtain coordinate points in no-ROI region
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/Get_grid_points.py ${sample}.svs ./ --level ${level} --mask_path ${sample}.No_Tumor.mask.npy

Based on preprocess, we generate a csv file(slide_lusc.csv) including slide,grids and targets.

##### train ResNet50 model
python /Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC/CTC_train_checkpoint.py --train_csv slide_lusc.csv --output /Data/yangml/Project/TCGA/Public7_TCGA_Part2/lung/data/LUSC/late_stage_lusc_slide/SVS/xml --nepochs 53 --checkpoint checkpoint_best.pth