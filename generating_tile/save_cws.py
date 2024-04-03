import generating_tile.cws_generator as cws
import os


def cws_all_process(obj):
    obj.generate_cws()
    obj.slide_thumbnail()
    obj.param()
    obj.final_scan_ini()
    obj.clust_tile_sh()
    obj.da_tile_sh()


def single_file_run(
        file_name, output_dir, input_dir, tif_obj=40,
        cws_objective_value=20, in_mpp=None,
        out_mpp=None, out_mpp_target_objective=40,
        parallel=False, **kwargs
):
    # print(file_name, flush=True)
    _, file_type = os.path.splitext(file_name)

    if file_type in ['.svs', '.ndpi', '.mrxs', '.kfb']:
        cws_obj = cws.CWSGENERATOR(
            output_dir=output_dir, file_name=file_name, input_dir=input_dir,
            cws_objective_value=cws_objective_value, in_mpp=in_mpp,
            out_mpp=out_mpp, out_mpp_target_objective=out_mpp_target_objective, parallel=parallel)
        cws_all_process(obj=cws_obj)

    if file_type == '.tif' or file_type == '.tiff' or file_type == '.png' or file_type == '.qptiff':
        cws_obj = cws.CWSGENERATOR(output_dir=output_dir, file_name=file_name, input_dir=input_dir,
                                   cws_objective_value=cws_objective_value, objective_power=tif_obj,
                                   in_mpp=in_mpp, out_mpp=out_mpp, out_mpp_target_objective=out_mpp_target_objective,
                                   parallel=parallel)
        cws_all_process(obj=cws_obj)


if __name__ == '__main__':
    pass
