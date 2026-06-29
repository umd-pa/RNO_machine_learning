import shutil
import argparse
import yaml
import os
from pathlib import Path

def check_yaml():
    ...
    #look inside folder
    #if no yaml found raise exception
    #else return yaml basedir

def main():
    # --- argparsing ---------------
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ice_model", choices=["greenland_simple_plus_nsigma", "greenland_simple_minus_nsigma", "greenland_simple_plus_z0sigma", "greenland_simple_minus_z0sigma", 
                                           "greenland_simple"], default="greenland_simple", required=False, help="Set of ice parameters to be used during simulation")
    p.add_argument("--signal_model", default="ARZ2020", required=False, help="Askaryan pulse model to be used during simulation")
    p.add_argument("--att_model", choices=["GL3", "GL3_plus_sigma", "GL3_minus_sigma"],
                    default="GL3", required=False, help="Signal attenuation model to be used during simulation")
    p.add_argument("--hw_resp", choices=[None, "gain_plus_sigma", "gain_minus_sigma"], default=None, required=False, help="Path to hardware response data with gain & phase per frequency")
    p.add_argument("--output_dir", default=check_yaml, help="Top-level output directory")
    args = p.parse_args()

    sim_params = ["ice_model", "signal_model", "att_model", "hw_resp"]
    is_benchmark = all(getattr(args, param) == p.get_default(param) for param in sim_params)

    multiple_changed_params=False
    if not is_benchmark:
        count=0
        for param in sim_params:
            if getattr(args, param) != p.get_default(param):
                sec_dir = param
                third_dir = os.path.basename(getattr(args, param)) if param == "hw_resp" else getattr(args, param)
                count+=1
        if count > 1:
            multiple_changed_params=True

    hw_path = "/data/condor_shared/users/nmanic/nuradio/myVENV/lib/python3.10/site-packages/NuRadioMC/NuRadioReco/detector/ARA/HardwareResponses/ARA_Electronics_TotalGain_TwoFilters.txt"
    if args.hw_resp == "gain_plus_sigma":
        hw_path = "/data/condor_shared/users/nmanic/nuradio/myVENV/lib/python3.10/site-packages/NuRadioMC/NuRadioReco/detector/ARA/HardwareResponses/ARA_Electronics_TotalGain_TwoFilters_plus_sigma.txt"
    elif args.hw_resp == "gain_minus_sigma":
        hw_path = "/data/condor_shared/users/nmanic/nuradio/myVENV/lib/python3.10/site-packages/NuRadioMC/NuRadioReco/detector/ARA/HardwareResponses/ARA_Electronics_TotalGain_TwoFilters_minus_sigma.txt"

    # --- directory setup ---------------
    if is_benchmark:
        main_dir = f"{args.output_dir}/classifier_consistency/{args.signal_model}/benchmark"
    elif multiple_changed_params:
        main_dir = f"{args.output_dir}/classifier_consistency/{args.signal_model}/other/{args.ice_model}-{args.att_model}"
    else:
        main_dir = f"{args.output_dir}/classifier_consistency/{args.signal_model}/{sec_dir}/{third_dir}"

    Path(main_dir).mkdir(parents=True, exist_ok=True)

    # --- create configs ---------------
    nu_config = {
        "noise": True, # specify if simulation should be run with or without noise.
        "sampling_rate": 5., # sampling rate in GHz used internally in the simulation.
        "speedup": {
            "minimum_weight_cut": 1.e-5,
            "delta_C_cut": 0.698 # 40 degree          
        },
        "propagation": {
            "ice_model": args.ice_model,
            "attenuation_model": args.att_model,
        },
        "signal": {
            "model": args.signal_model,
        },
        "trigger": {
            "noise_temperature": 300 # in Kelvin.
        },
        "weights": {
            "weight_mode": "core_mantle_crust"
        }
    }

    noise_config = {
        "noise": True, # specify if simulation should be run with or without noise.
        "sampling_rate": 5., # sampling rate in GHz used internally in the simulation.
        "speedup": {
            "minimum_weight_cut": 1.e-5,
            "delta_C_cut": 0.698 # 40 degree          
        },
        "propagation": {
            "ice_model": args.ice_model,
            "attenuation_model": args.att_model,
        },
        "signal": {
            "model": args.signal_model,
            "zerosignal": True
        },
        "trigger": {
            "noise_temperature": 300 # in Kelvin.
        },
        "weights": {
            "weight_mode": "core_mantle_crust"
        }
    }

    with open(f"{main_dir}/nu_config.yaml", "w") as f:
        yaml.dump(nu_config, f, default_flow_style=False, sort_keys=False)
    with open(f"{main_dir}/noise_config.yaml", "w") as f:
        yaml.dump(noise_config, f, default_flow_style=False, sort_keys=False)

    # --- submit jobs ---------------

if __name__ == "__main__":
    main()