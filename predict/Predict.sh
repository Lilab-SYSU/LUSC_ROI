#!/usr/bin/env bash
python3 /home/myang/PycharmProjects/PyTorch_test/Pathology_WSI/LILI/predict/Get_grid_points.py L14733-C.kfb.svs ./ --level 4
python3 /home/myang/PycharmProjects/PyTorch_test/Pathology_WSI/LILI/predict/Hep_test.py --test_csv slide.csv --output ./ --model ../checkpoint_best.pth --batch_size 256
####for input image normaled
python3 /home/myang/PycharmProjects/PyTorch_test/Pathology_WSI/LILI/predict/Hep_test.py --test_csv slide.csv --output ./ --model ../checkpoint_best.pth --batch_size 256 --targetImage /public5/lilab/student/myang/project/lili/GG/N02171-A.kfb.mask.Tumor/59712_45621.jpg --colornorm
###example csv /public5/lilab/student/myang/project/lili/GG/slide22.csv