from pathlib import Path
import h5py
import numpy as np
from amuse.units import units
from amuse.lab import Particles
import pandas as pd

# ---------------------------------------------------------------------------
# LOAD THE INITIAL CONDITION FROM TXT FILES
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# LOAD THE INITIAL CONDITION FROM PRECOMPUTED TXT FILES
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# WRITE THE RESULT INTO HDF5 FILES
# ---------------------------------------------------------------------------
def save_to_hdf5(results, params, output_path):
    tmp_path = str(output_path) + ".tmp"

    r_star_val = params["R_STAR"]
    if hasattr(r_star_val, "value_in"):
        r_star_val = r_star_val.value_in(units.RSun)

    with h5py.File(tmp_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["M_STAR_MSun"] = params["M_STAR"]
        meta.attrs["R_STAR_RSun"] = r_star_val
        meta.attrs["T_FINAL_DAYS"] = params["T_FINAL_DAYS"]
        meta.attrs["N_POINTS_PER_TRANSIT"] = params["N_POINTS_PER_TRANSIT"]
        meta.attrs["COARSE_FACTOR"] = params["COARSE_FACTOR"]
        meta.attrs["BUFFER_FACTOR"] = params["BUFFER_FACTOR"]
        meta.attrs["MAX_COARSE_STEP_DAYS"] = params["MAX_COARSE_STEP_DAYS"]
        meta.attrs["ENERGY_ERROR_THRESHOLD"] = params["ENERGY_ERROR_THRESHOLD"]
        meta.attrs["n_steps_saved"] = len(results["time"])
        meta.attrs["runtime_seconds"] = results.get("runtime_seconds", -1.0)
        meta.attrs["max_energy_error"] = results.get("max_energy_error", -1.0)

        f.create_dataset("time_days", data=np.asarray(results["time"]), compression="gzip")

        star_grp = f.create_group("star")
        star_grp.attrs["mass_MSun"] = results["star_mass_MSun"]
        star_grp.attrs["radius_RSun"] = results["star_radius_RSun"]
        star_grp.create_dataset("x_au", data=np.asarray(results["star_x_au"]), compression="gzip")
        star_grp.create_dataset("y_au", data=np.asarray(results["star_y_au"]), compression="gzip")
        star_grp.create_dataset("vx_kms", data=np.asarray(results["star_vx_kms"]), compression="gzip")
        star_grp.create_dataset("vy_kms", data=np.asarray(results["star_vy_kms"]), compression="gzip")

        sys_grp = f.create_group("system")
        sys_grp.attrs["energy_unit"] = "Joule"
        sys_grp.create_dataset("energy_total_J", data=np.asarray(results["energy_total_J"]), compression="gzip")
        sys_grp.create_dataset("energy_rel_error", data=np.asarray(results["energy_rel_error"]), compression="gzip")

        planets_grp = f.create_group("planets")
        for name in results["planet_names"]:
            pg = planets_grp.create_group(name)
            pg.attrs["mass_MSun"] = results["planet_mass_MSun"][name]
            pg.attrs["radius_REarth"] = results["planet_radius_REarth"][name]
            pg.create_dataset("dx_au", data=np.asarray(results["planet_dx_au"][name]), compression="gzip")
            pg.create_dataset("dy_au", data=np.asarray(results["planet_dy_au"][name]), compression="gzip")
            pg.create_dataset("x_au", data=np.asarray(results["planet_x_au"][name]), compression="gzip")
            pg.create_dataset("y_au", data=np.asarray(results["planet_y_au"][name]), compression="gzip")
            pg.create_dataset("vx_kms", data=np.asarray(results["planet_vx_kms"][name]), compression="gzip")
            pg.create_dataset("vy_kms", data=np.asarray(results["planet_vy_kms"][name]), compression="gzip")

    Path(tmp_path).replace(output_path)


# ---------------------------------------------------------------------------
# STORE THE RESULT
# ---------------------------------------------------------------------------
def package_results(var_time, star, star_x, star_y, star_vx, star_vy,
                      planet_names, planets, planet_x, planet_y, planet_dx, planet_dy,
                      planet_vx, planet_vy, energy_error_history, energy_total_J_history):
    results = {
        "time": var_time,
        "star_mass_MSun": star.mass.value_in(units.MSun),
        "star_radius_RSun": star.radius.value_in(units.RSun),
        "star_x_au": star_x, "star_y_au": star_y,
        "star_vx_kms": star_vx, "star_vy_kms": star_vy,
        "energy_total_J": energy_total_J_history,
        "energy_rel_error": energy_error_history,
        "planet_names": planet_names,
        "planet_mass_MSun": {name: planets[i].mass.value_in(units.MSun) for i, name in enumerate(planet_names)},
        "planet_radius_REarth": {name: planets[i].radius.value_in(units.REarth) for i, name in enumerate(planet_names)},
        "planet_x_au": planet_x, "planet_y_au": planet_y,
        "planet_dx_au": planet_dx, "planet_dy_au": planet_dy,
        "planet_vx_kms": planet_vx, "planet_vy_kms": planet_vy
    }
    return results

# ---------------------------------------------------------------------------
# READ THE HDF5 FILES TO PANDAS DATAFRAME
# ---------------------------------------------------------------------------
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