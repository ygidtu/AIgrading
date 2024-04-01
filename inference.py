#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
from argparse import ArgumentParser, ArgumentError
from glob import glob
from multiprocessing import Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tqdm import tqdm

from generating_tile.save_cws import single_file_run
from inference_slide.predict_gp import generate_gp
from inference_slide.ss1_final import ss1_final
from inference_slide.ss1_stich import ss1_stich


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


def call(args):
    try:
        single_file_run(**args)
    except ZeroDivisionError:
        pass


def inference_slide(args):
    generate_gp(**args["gp"])

    # step2: stich to ss1 level
    ss1_stich(**args["stich"])

    # step3: post-processing
    ss1_final(**args["final"])


def main():
    try:
        args = parser.parse_args(sys.argv[1:])
    except ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)

    output_dir = args.save_dir
    tiling_dir = os.path.join(output_dir, "cws_tiling")
    os.makedirs(tiling_dir, exist_ok=True)

    generated_gp = os.path.join(output_dir, "generated_gp")
    os.makedirs(generated_gp, exist_ok=True)

    stich_dir = os.path.join(output_dir, "stich")
    os.makedirs(stich_dir, exist_ok=True)

    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    wsi = sorted(find_files(args.data_dir, args.file_name_pattern))

    cmds = []
    for wsi_i in wsi:

        if glob(os.path.join(tiling_dir, os.path.basename(wsi_i), "Ss1.jpg")):
            continue

        cmds.append({
            'output_dir': tiling_dir,
            'input_dir': os.path.dirname(wsi_i),
            'file_name': os.path.basename(wsi_i),
            'wsi_input': wsi_i,
            'tif_obj': 40,
            'cws_objective_value': 20,
            'in_mpp': None,
            'out_mpp': args.output_mpp,
            'out_mpp_target_objective': 40,
            'parallel': False
        })

    with Pool(args.n_jobs) as p:
        list(tqdm(p.imap(call, cmds), total=len(cmds)))

    cws_files = sorted(glob(os.path.join(tiling_dir, "*")))
    cmds = []
    for cws_file in cws_files:
        cmds.append({
            "gp": {
                "datapath": cws_file, "save_dir": generated_gp,
                "color_norm": args.color_norm, "patch_size": 768,
                "patch_stride": 192, "nClass": 7
            },
            "stich": {
                "cws_file": cws_file, "annotated_dir": generated_gp,
                "output_dir": stich_dir
            },
            "final": {
                "cws_file": cws_file, "final_dir": final_dir,
                "stich_dir": stich_dir
            }
        })

    with Pool(args.n_jobs) as p:
        list(tqdm(p.imap_unordered(inference_slide, cmds), total=len(cmds)))


if __name__ == '__main__':
    main()
