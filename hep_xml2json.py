#-*-coding:utf-8-*-
import json
import xml.etree.ElementTree as ET
import copy
import os
import argparse
import numpy as np
# os.chdir('/public5/lilab/student/myang/project/lili/GG')
parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the output annotation in json format')
# root = ET.parse('N00733-B.kfb.xml').getroot()
#
# annotations_tumor = root.findall('./Annotations/Annotation[@PartOfGroup="Ca"]')
# annotations_normal = root.findall
# X = list(map(lambda x: float(x.get('X')),
#                      annotations_tumor[0].findall('./Coordinates/Coordinate')))
# Y = list(map(lambda x: float(x.get('X')),
#                      annotations_tumor[0].findall('./Coordinates/Coordinate')))
class Formatter(object):
    """
    Format converter e.g. CAMELYON16 to internal json
    """
    def camelyon16xml2json(inxml, outjson):
        """
        Convert an annotation of camelyon16 xml format into a json format.
        Arguments:
            inxml: string, path to the input camelyon16 xml format
            outjson: string, path to the output json format
        """
        root = ET.parse(inxml).getroot()
        annotations_tumor = root.findall('./Annotations/Annotation[@PartOfGroup="None"]')
        #annotations_normal = root.findall('./Annotations/Annotation[@PartOfGroup="PN"]')
        # if annotations_normal:
        #     print('PN is normal')
        # else:
        #     annotations_normal = root.findall('./Annotations/Annotation[@PartOfGroup="N"]')
        #     if annotations_normal:
        #         print('N is normal')
        # annotations_negative = annotations_2

        json_dict = {}
        json_dict['Normal'] = []
        json_dict['Tumor'] = []

        for annotation in annotations_tumor:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['Tumor'].append({'name': name, 'vertices': vertices})

        # for annotation in annotations_normal:
        #     X = list(map(lambda x: float(x.get('X')),
        #              annotation.findall('./Coordinates/Coordinate')))
        #     Y = list(map(lambda x: float(x.get('Y')),
        #              annotation.findall('./Coordinates/Coordinate')))
        #     vertices = np.round([X, Y]).astype(int).transpose().tolist()
        #     name = annotation.attrib['Name']
        #     json_dict['Normal'].append({'name': name, 'vertices': vertices})
        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)
        return json_dict


# annotation_tumor[0].attrib
####annotation test###
args = parser.parse_args()
# annot_dict = Formatter.camelyon16xml2json(args.xml_path,args.json_path)

annot_dict = Formatter.camelyon16xml2json(args.xml_path,args.json_path)

# import numpy as np
# import openslide
# import cv2
# import json
# level=3
# slide = openslide.OpenSlide('N02171-A.kfb.svs')
# w, h = slide.level_dimensions[level]
# mask_tumor = np.zeros((h, w))
# factor = slide.level_downsamples[level]
# tumor_polygons = annot_dict['Tumor']
# for tumor_polygon in tumor_polygons:
#     # plot a polygon
#     name = tumor_polygon["name"]
#     vertices = np.array(tumor_polygon["vertices"]) / factor
#     vertices = vertices.astype(np.int32)
#
#     cv2.fillPoly(mask_tumor, [vertices], (255))
#     mask_tumor = mask_tumor[:] > 127