#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import pickle
import re
from glob import glob

import cv2
import numpy as np


def ss1_stich(cws_file, annotated_dir, output_dir):
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)

    def natural_key(string_):
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

    wsi_name = os.path.split(cws_file)[-1]

    if os.path.exists(os.path.join(output_dir, wsi_name + "_Ss1.png")):
        print(wsi_name, 'exists')
        pass
    else:
        param = pickle.load(open(os.path.join(cws_file, 'param.p'), 'rb'))
        ss1 = cv2.imread(os.path.join(cws_file, 'Ss1.jpg'))
        slide_dimension = np.array(param['slide_dimension']) / param['rescale']
        slide_w, slide_h = slide_dimension
        cws_w, cws_h = param['cws_read_size']
        divisor_w = np.ceil(slide_w / cws_w)

        w, h = int(slide_w * 0.0625), int(slide_h * 0.0625)
        img_all = np.zeros((max(h, ss1.shape[0]) + 1, max(ss1.shape[1], w) + 1, 3))

        drivepath, imagename = os.path.split(cws_file)
        annotated_dir_slide = os.path.join(annotated_dir, imagename)
        annotated_path = annotated_dir_slide
        images = sorted(glob(os.path.join(annotated_path, 'Da*')), key=natural_key)

        for ii in images:
            ii = os.path.basename(ii)
            cws_i = int(re.search(r'\d+', ii).group())
            h_i_ori = int(np.floor(cws_i / divisor_w)) * cws_h
            w_i = int((cws_i - h_i_ori / cws_h * divisor_w)) * int(cws_w * 0.0625)
            h_i = int(np.floor(cws_i / divisor_w)) * int(cws_h * 0.0625)
            img = cv2.imread(os.path.join(annotated_path, ii))

            w_r = max(int(img.shape[1] * 0.0625), 1)
            h_r = max(int(img.shape[0] * 0.0625), 1)
            img_r = cv2.resize(img, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
            img_all[h_i: h_i + int(img_r.shape[0]), w_i: w_i + int(img_r.shape[1]), :] = img_r
            cv2.imwrite(os.path.join(output_dir, imagename + "_Ss1.png"), img_all)


if __name__ == "__main__":
    pass
