from pathlib import Path
import h5py
import numpy as np
from amuse.units import units
from amuse.lab import Particles
import pandas as pd
from .memory import StreamingHDF5Writer
from . import schema

def load_input_file(input_path):
    code = Path(input_path).read_text()
    namespace = {}

    exec(compile(code, str(input_path), "exec"), namespace)

    if "M_STAR" not in namespace or "R_STAR" not in namespace:
        raise ValueError(f"Input file '{input_path}' must define M_STAR and R_STAR.")

    planets_list = namespace.get("planets", namespace.get("PLANETS"))
    if planets_list is None:
        raise ValueError(
            f"Input file '{input_path}' must define a planet list as the variable 'planets'or 'PLANETS'"
        )

    required_keys = {"name", "mass", "radius", "semi_major_axis", "ecc", "true_anomaly", "inc", "loa", "aop"}
    for idx, p in enumerate(planets_list):
        missing = required_keys - set(p.keys())
        if missing:
            raise ValueError(
                f"Planet #{idx} ('{p.get('name', '?')}') in '{input_path}' is missing required field(s): {missing}"
            )

    defaults = {
        "T_FINAL_DAYS": 500.0,
        "N_POINTS_PER_TRANSIT": int(1e2),
        "COARSE_FACTOR": 100,
        "BUFFER_FACTOR": 5.0,
        "MAX_COARSE_STEP_DAYS": 1.0,
        "MIN_TIMESTEP_DAYS": 1.0 / 86400.0,
        "ENERGY_ERROR_THRESHOLD": 1e-8,
        "STAR_POSITION": [0.0, 0.0, 0.0],
        "STAR_VELOCITY": [0.0, 0.0, 0.0],
        "N_WORKERS": 2,
        "GRAVITY_TIMESTEP_PARAM": 1e-14,
    }

    params = {"M_STAR": namespace["M_STAR"], "R_STAR": namespace["R_STAR"], "planets_list": planets_list}
    for key, default_val in defaults.items():
        if key in namespace:
            params[key] = namespace[key]
        else:
            params[key] = default_val
            print(f"[INFO] '{key}' not found in '{input_path}', using default: {default_val}")

    params["Output_Dir"] = namespace.get("Output_Dir", ".")
    params["Output_file_name"] = namespace.get("Output_file_name", None)

    return params

def open_streaming_writer(output_path, params, star, planet_names, planets):
    return StreamingHDF5Writer(output_path, params, star, planet_names, planets)

def load_precomputed_input_file(input_path):
    code = Path(input_path).read_text()
    namespace = {}

    exec(compile(code, str(input_path), "exec"), namespace)

    if "M_STAR" not in namespace or "R_STAR" not in namespace:
        raise ValueError(f"Input file '{input_path}' must define M_STAR and R_STAR.")

    planets_list = namespace.get("planets", namespace.get("PLANETS"))
    if planets_list is None:
        raise ValueError(
            f"Input file '{input_path}' must define a planet list as the variable 'planets'or 'PLANETS'"
        )

    required_keys = {"name", "mass", "radius", "x_au", "y_au", "z_au", "vx_au_yr", "vy_au_yr", "vz_au_yr"}
    for idx, p in enumerate(planets_list):
        missing = required_keys - set(p.keys())
        if missing:
            raise ValueError(
                f"Planet #{idx} ('{p.get('name', '?')}') in '{input_path}' is missing required field(s): {missing}"
            )

    defaults = {
        "T_FINAL_DAYS": 500.0,
        "N_POINTS_PER_TRANSIT": int(1e2),
        "COARSE_FACTOR": 100,
        "BUFFER_FACTOR": 5.0,
        "MAX_COARSE_STEP_DAYS": 1.0,
        "MIN_TIMESTEP_DAYS": 1.0 / 86400.0,
        "ENERGY_ERROR_THRESHOLD": 1e-8,
        "STAR_POSITION": [0.0, 0.0, 0.0],
        "STAR_VELOCITY": [0.0, 0.0, 0.0],
        "N_WORKERS": 2,
        "GRAVITY_TIMESTEP_PARAM": 1e-14,
    }

    params = {"M_STAR": namespace["M_STAR"], "R_STAR": namespace["R_STAR"], "planets_list": planets_list}
    for key, default_val in defaults.items():
        if key in namespace:
            params[key] = namespace[key]
        else:
            params[key] = default_val
            print(f"[INFO] '{key}' not found in '{input_path}', using default: {default_val}")

    params["Output_Dir"] = namespace.get("Output_Dir", ".")
    params["Output_file_name"] = namespace.get("Output_file_name", None)

    return params

def save_to_hdf5(results, params, output_path):
    tmp_path = str(output_path) + ".tmp"

    r_star_val = schema.resolve_r_star_au_value(params["R_STAR"])

    with h5py.File(tmp_path, "w") as f:
        meta = f.create_group("metadata")
        schema.write_static_metadata(meta, params, r_star_val)
        schema.write_runtime_metadata(
            meta,
            n_steps_saved=len(results["time"]),
            runtime_seconds=results.get("runtime_seconds", -1.0),
            max_energy_error=results.get("max_energy_error", -1.0),
        )

        f.create_dataset("time_days", data=np.asarray(results["time"]), compression="gzip")

        star_grp = f.create_group("star")
        star_grp.attrs["mass_MSun"] = results["star_mass_MSun"]
        star_grp.attrs["radius_RSun"] = results["star_radius_RSun"]
        star_source_keys = {
            "x_au": "star_x_au", "y_au": "star_y_au", "z_au": "star_z_au",
            "vx_kms": "star_vx_kms", "vy_kms": "star_vy_kms", "vz_kms": "star_vz_kms",
        }
        for ds_name in schema.STAR_DATASET_NAMES:
            star_grp.create_dataset(
                ds_name, data=np.asarray(results[star_source_keys[ds_name]]), compression="gzip"
            )

        sys_grp = f.create_group("system")
        sys_grp.attrs["energy_unit"] = "Joule"
        system_source_keys = {"energy_total_J": "energy_total_J", "energy_rel_error": "energy_rel_error"}
        for ds_name in schema.SYSTEM_DATASET_NAMES:
            sys_grp.create_dataset(
                ds_name, data=np.asarray(results[system_source_keys[ds_name]]), compression="gzip"
            )

        planets_grp = f.create_group("planets")
        planet_source_keys = {
            "dx_au": "planet_dx_au", "dy_au": "planet_dy_au", "dz_au": "planet_dz_au",
            "x_au": "planet_x_au", "y_au": "planet_y_au", "z_au": "planet_z_au",
            "vx_kms": "planet_vx_kms", "vy_kms": "planet_vy_kms", "vz_kms": "planet_vz_kms",
        }
        for name in results["planet_names"]:
            pg = planets_grp.create_group(name)
            pg.attrs["mass_MSun"] = results["planet_mass_MSun"][name]
            pg.attrs["radius_REarth"] = results["planet_radius_REarth"][name]
            for ds_name in schema.PLANET_DATASET_NAMES:
                pg.create_dataset(
                    ds_name,
                    data=np.asarray(results[planet_source_keys[ds_name]][name]),
                    compression="gzip",
                )

    Path(tmp_path).replace(output_path)

def package_results(var_time, star, star_x, star_y, star_z, star_vx, star_vy, star_vz,
                      planet_names, planets,
                      planet_x, planet_y, planet_z,
                      planet_dx, planet_dy, planet_dz,
                      planet_vx, planet_vy, planet_vz,
                      energy_error_history, energy_total_J_history):
    results = {
        "time": var_time,
        "star_mass_MSun": star.mass.value_in(units.MSun),
        "star_radius_RSun": star.radius.value_in(units.RSun),
        "star_x_au": star_x, "star_y_au": star_y, "star_z_au": star_z,
        "star_vx_kms": star_vx, "star_vy_kms": star_vy, "star_vz_kms": star_vz,
        "energy_total_J": energy_total_J_history,
        "energy_rel_error": energy_error_history,
        "planet_names": planet_names,
        "planet_mass_MSun": {name: planets[i].mass.value_in(units.MSun) for i, name in enumerate(planet_names)},
        "planet_radius_REarth": {name: planets[i].radius.value_in(units.REarth) for i, name in enumerate(planet_names)},
        "planet_x_au": planet_x, "planet_y_au": planet_y, "planet_z_au": planet_z,
        "planet_dx_au": planet_dx, "planet_dy_au": planet_dy, "planet_dz_au": planet_dz,
        "planet_vx_kms": planet_vx, "planet_vy_kms": planet_vy, "planet_vz_kms": planet_vz
    }
    return results

def read_result(file_dir):
    dataframes = {}
    attributes = {}

    with h5py.File(file_dir, 'r') as f:
        planets_group = f['planets']
        for p_name in planets_group.keys():
            p_group = planets_group[p_name]
            
            planet_data = {}
            for ds_name in p_group.keys():
                planet_data[ds_name] = p_group[ds_name][()]
            
            dataframes[p_name] = pd.DataFrame(planet_data)
            attributes[p_name] = dict(p_group.attrs)
            
        star_group = f['star']
        star_data = {}
        for ds_name in star_group.keys():
            star_data[ds_name] = star_group[ds_name][()]
        
        dataframes['star'] = pd.DataFrame(star_data)
        attributes['star'] = dict(star_group.attrs)

        time_data = f['time_days'][()]
        dataframes['time'] = pd.DataFrame({'time_days': time_data})

        system_group = f['system']
        system_data = {}
        for ds_name in system_group.keys():
            system_data[ds_name] = system_group[ds_name][()]

        dataframes['system'] = pd.DataFrame(system_data)

        meta_group = f['metadata']
        dataframes['metadata'] = dict(meta_group.attrs)

    dataframes['attributes'] = attributes

    return dataframes
