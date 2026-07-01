import argparse
import yaml
import os
from pathlib import Path

def check_yaml():
    proj_dir = Path(__file__).resolve().parent.parent
    yaml_path = proj_dir / 'user_config.yaml'
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
        return yaml_content

def get_abs_path(rel_path):
    """
    Converts a relative path to an absolute path based on 
    the location of THIS script (create_dagman.py).
    """
    # This gets the folder where create_dagman.py lives (e.g., .../jobs/)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, rel_path))

def main():
    # --- argparsing ---------------
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--venv_path", required=True, help="path to user's venv file")
    p.add_argument("--ice_model", choices=["greenland_simple_plus_nsigma", "greenland_simple_minus_nsigma", "greenland_simple_plus_z0sigma", "greenland_simple_minus_z0sigma", 
                                           "greenland_simple"], default="greenland_simple", required=False, help="Set of ice parameters to be used during simulation")
    p.add_argument("--signal_model", default="ARZ2020", required=False, help="Askaryan pulse model to be used during simulation")
    p.add_argument("--att_model", choices=["GL3", "GL3_plus_sigma", "GL3_minus_sigma"],
                    default="GL3", required=False, help="Signal attenuation model to be used during simulation")
    p.add_argument("--hw_resp", choices=[None, "gain_plus_sigma", "gain_minus_sigma"], default=None, required=False, help="Path to hardware response data with gain & phase per frequency")
    p.add_argument("--data_dir", default=None, help="Top-level data directory")
    args = p.parse_args()

    if args.data_dir is None:
        try:
            data_dir = check_yaml()['data_dir']
        except Exception as e:
            p.error(f'Failed to load default data_dir from Yaml: {e}')
    else:
        data_dir = args.data_dir

    data_dir = os.path.abspath(data_dir)

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

    # --- directory setup ---------------
    if is_benchmark:
        main_dir = f"{data_dir}/{args.signal_model}/benchmark"
    elif multiple_changed_params:
        main_dir = f"{data_dir}/{args.signal_model}/other/{args.ice_model}-{args.att_model}-{args.hw_resp}"
    else:
        main_dir = f"{data_dir}/{args.signal_model}/{sec_dir}/{third_dir}"

    nu_dir = f'{main_dir}/nu/nur'
    noise_dir = f'{main_dir}/noise/nur'
    nu_logs_dir = f'{main_dir}/nu/logs'
    noise_logs_dir = f'{main_dir}/noise/logs'
    Path(main_dir).mkdir(parents=True, exist_ok=True)
    Path(nu_dir).mkdir(parents=True, exist_ok=True)
    Path(noise_dir).mkdir(parents=True, exist_ok=True)
    Path(nu_logs_dir).mkdir(parents=True, exist_ok=True)
    Path(noise_logs_dir).mkdir(parents=True, exist_ok=True)

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

    # --- paths ---------------
    hw_path = "/data/condor_shared/users/nmanic/nuradio/myVENV/lib/python3.10/site-packages/NuRadioMC/NuRadioReco/detector/ARA/HardwareResponses/ARA_Electronics_TotalGain_TwoFilters.txt"
    if args.hw_resp == "gain_plus_sigma":
        hw_path = "/data/condor_shared/users/nmanic/nuradio/myVENV/lib/python3.10/site-packages/NuRadioMC/NuRadioReco/detector/ARA/HardwareResponses/ARA_Electronics_TotalGain_TwoFilters_plus_sigma.txt"
    elif args.hw_resp == "gain_minus_sigma":
        hw_path = "/data/condor_shared/users/nmanic/nuradio/myVENV/lib/python3.10/site-packages/NuRadioMC/NuRadioReco/detector/ARA/HardwareResponses/ARA_Electronics_TotalGain_TwoFilters_minus_sigma.txt"

    generate_dir = Path(__file__).resolve().parent

    nu_sub              = f"{generate_dir}/nu_sims.sub"
    noise_sub           = f"{generate_dir}/noise_sims.sub"
    nu_extract_sub         = f"{generate_dir}/nu_extract.sub"
    noise_extract_sub         = f"{generate_dir}/noise_extract.sub"
    station_path        = f"{generate_dir}/station.json"
    sim_script_path          = f"{generate_dir}/simulate.py"
    extract_script_path        = f"{generate_dir}/extract.py"
    nu_config_path      = f"{main_dir}/nu_config.yaml"
    noise_config_path   = f"{main_dir}/noise_config.yaml"
    neutrino_dir           = f"{data_dir}/neutrinos/"

    # --- create dagman ---------------
    if is_benchmark:
        dag_filename = f'{main_dir}/benchmark.dag'
    elif multiple_changed_params:
        dag_filename = f'{main_dir}/{args.ice_model}-{args.att_model}-{args.hw_resp}.dag'
    else:
        dag_filename = f'{main_dir}/{third_dir}.dag'

    with open(dag_filename, 'w') as f:
        f.write("# HTCondor DAG file\n")

        f.write(f"JOB nu_sim {nu_sub}\n")
        f.write(f'VARS nu_sim in_dir="{neutrino_dir}" output_base="{main_dir}" venv="{args.venv_path}" step2="{sim_script_path}" station="{station_path}" config="{nu_config_path}" hw_path="{hw_path}"\n')

        f.write(f"JOB noise_sim {noise_sub}\n")
        f.write(f'VARS noise_sim in_dir="{neutrino_dir}" output_base="{main_dir}" venv="{args.venv_path}" step2="{sim_script_path}" station="{station_path}" config="{noise_config_path}" hw_path="{hw_path}"\n')

        f.write(f"JOB extract_nu {nu_extract_sub}\n")
        f.write(f'VARS extract_nu in_dir=\"{nu_dir}/*.nur\" output_base="{main_dir}" venv="{args.venv_path}" extract_path="{extract_script_path}"\n')

        f.write(f"JOB extract_noise {noise_extract_sub}\n")
        f.write(f'VARS extract_noise in_dir=\"{noise_dir}/*.nur\" output_base="{main_dir}" venv="{args.venv_path}" extract_path="{extract_script_path}"\n')

        f.write("PARENT nu_sim CHILD extract_nu\n")
        f.write("PARENT noise_sim CHILD extract_noise\n")

    print(f'Dag file created. Please submit with condor_submit_dag -F {dag_filename}')
if __name__ == "__main__":
    main()