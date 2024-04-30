#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
from argparse import ArgumentParser, ArgumentError
from glob import glob
from multiprocessing import Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from loguru import logger
from PIL import Image
# from tensorflow.python.client import device_lib
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

from generating_tile.save_cws import single_file_run
from inference_slide.predict_gp import generate_gp

parser = ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='path to raw image data')
parser.add_argument('-mpp', '--output_mpp', dest='output_mpp', help='output magnification', default=0.22, type=float)
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='path to save all output files', default=None)
parser.add_argument('-c', '--color', dest='color_norm', help='color normalization', action='store_false')
parser.add_argument('-n', '--nfile', dest='nth_file', help='the n-th file', default=0, type=int)
parser.add_argument('-j', '--n_jobs', help='the n-th processes to use', default=4, type=int)


def find_files(path, pattern):
    fs = []
    for parent, _, files in os.walk(path):
        for file in files:
            if file.endswith(pattern):
                fs.append(os.path.join(parent, file))
    return fs


def load_image(args):
    key, path, width, height = args
    img = Image.open(path)
    img = img.resize((width, height))
    return {key: np.array(img)}


def concat_image(image_coords, tiling_dir: str, max_h, max_w, postfix: str = ".jpg", n_jobs: int = 10):
    cmds = []
    for key, value in image_coords.items():
        cmds.append([key, os.path.join(tiling_dir, key + postfix), value["width"], value["height"]])

    res = {}
    with Pool(n_jobs) as p:
        for row in list(tqdm(p.imap(load_image, cmds), total=len(cmds), desc="Load images")):
            res.update(row)

    image = np.zeros([max_h, max_w, 3], np.uint8)
    for key, value in tqdm(image_coords.items(), desc="Concat images"):
        image[value["start_h"]:value["end_h"], value["start_w"]:value["end_w"], :] = res[key]

    return image


def merge_images(input_file, generated_dir: str, output_path: str, n_jobs: int = 10, output_width: int = 2000):
    max_w, max_h = 0, 0
    image_coords = {}
    with open(input_file) as r:
        for line in r:
            line = line.split()
            key = line[0].strip().replace(":", "")
            image_coords[key] = {}

            for row in line[1:]:
                x, y = row.split(":")
                image_coords[key][x.strip()] = int(y.replace(",", ""))

            max_w = max(max_w, image_coords[key]["end_w"])
            max_h = max(max_h, image_coords[key]["end_h"])

    tilling_img = concat_image(image_coords, os.path.dirname(input_file), max_h, max_w, postfix=".jpg", n_jobs=n_jobs)
    tilling_img = Image.fromarray(tilling_img)
    # tilling_img.save(os.path.join(output_path, "tiling_image.png"))

    generated_img = concat_image(image_coords, generated_dir, max_h, max_w, postfix=".png", n_jobs=n_jobs)
    generated_img = Image.fromarray(generated_img)
    # generated_img.save(os.path.join(output_path, "generated_image.png"))

    resize = [output_width, int(max_h * (output_width / max_w))]

    tilling_img = tilling_img.resize(resize)
    generated_img = generated_img.resize(resize)

    new_img = Image.blend(tilling_img, generated_img, 0.5)
    new_img.save(output_path, "PNG")


def inference_slide(args):
    generate_gp(**args)

    # # step2: stich to ss1 level
    # ss1_stich(**args["stich"])
    #
    # # step3: post-processing
    # ss1_final(**args["final"])


def main():
    try:
        args = parser.parse_args(sys.argv[1:])
    except ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)

    # print(device_lib.list_local_devices())
    output_dir = args.save_dir
    tiling_dir = os.path.join(output_dir, "cws_tiling")
    os.makedirs(tiling_dir, exist_ok=True)

    generated_gp = os.path.join(output_dir, "generated_gp")
    os.makedirs(generated_gp, exist_ok=True)

    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    wsi = sorted(find_files(args.data_dir, args.file_name_pattern))

    for wsi_i in tqdm(wsi, desc="Generating tiles..."):

        if glob(os.path.join(tiling_dir, os.path.basename(wsi_i), "Ss1.jpg")):
            continue

        try:
            single_file_run(**{
                'output_dir': tiling_dir,
                'input_dir': os.path.dirname(wsi_i),
                'file_name': os.path.basename(wsi_i),
                'wsi_input': wsi_i,
                'tif_obj': 40,
                'cws_objective_value': 20,
                'in_mpp': None,
                'out_mpp': args.output_mpp,
                'out_mpp_target_objective': 40,
                'parallel': args.n_jobs,
                "generated_dir": generated_gp
            })
        except ZeroDivisionError as err:
            logger.error(err)
            continue

    cws_files = sorted(glob(os.path.join(tiling_dir, "*")))
    for cws_file in tqdm(cws_files, desc="Predict slides..."):
        inference_slide({
            "datapath": cws_file, "save_dir": generated_gp,
            "color_norm": args.color_norm, "patch_size": 768,
            "patch_stride": 192, "nClass": 7, "tiling_dir": tiling_dir
        })

    for wsi_i in tqdm(wsi, desc="Merging images..."):
        merge_images(
            os.path.join(tiling_dir, os.path.basename(wsi_i), "Output.txt"),
            os.path.join(generated_gp, os.path.basename(wsi_i)),
            output_path=os.path.join(str(final_dir), os.path.basename(wsi_i) + ".png"),
            n_jobs=args.n_jobs
        )


if __name__ == '__main__':
    main()
