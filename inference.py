#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
from argparse import ArgumentParser, ArgumentError
from glob import glob
from multiprocessing import Pool

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


def call(args):
    try:
        single_file_run(**args)
    except ZeroDivisionError:
        pass


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

    final_output_dir = os.path.join(output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)

    ss1_dir = os.path.join(output_dir, "ss1")
    os.makedirs(ss1_dir, exist_ok=True)

    ss1_final_dir = os.path.join(output_dir, "ss1_final")
    os.makedirs(ss1_final_dir, exist_ok=True)

    print("running tiles")
    wsi = sorted(glob(os.path.join(args.data_dir, args.file_name_pattern)))

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

    ######step0: generate cws tiles from single-cell pipeline
    ######step1: generate growth pattern for tiles
    generate_gp(datapath=tiling_dir,
                save_dir=final_output_dir,
                file_pattern=args.file_name_pattern,
                color_norm=args.color_norm, nfile=args.nth_file,
                patch_size=768, patch_stride=192, nClass=7)

    #######step2: stich to ss1 level
    ss1_stich(cws_folder=tiling_dir,
              annotated_dir=args.save_dir,
              output_dir=ss1_dir,
              nfile=args.nth_file,
              file_pattern=args.file_name_pattern)

    #######step3: post-processing
    ss1_final(cws_folder=tiling_dir,
              ss1_dir=ss1_dir,
              ss1_final_dir=ss1_final_dir,
              nfile=args.nth_file,
              file_pattern=args.file_name_pattern)


if __name__ == '__main__':
    main()
