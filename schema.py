from amuse.units import units


METADATA_KEYS = (
    "M_STAR_MSun",
    "R_STAR_RSun",
    "T_FINAL_DAYS",
    "N_POINTS_PER_TRANSIT",
    "COARSE_FACTOR",
    "BUFFER_FACTOR",
    "MAX_COARSE_STEP_DAYS",
    "ENERGY_ERROR_THRESHOLD",
)

METADATA_RUNTIME_KEYS = (
    "n_steps_saved",
    "runtime_seconds",
    "max_energy_error",
)

STAR_DATASET_NAMES = (
    "x_au", "y_au", "z_au",
    "vx_kms", "vy_kms", "vz_kms",
)

SYSTEM_DATASET_NAMES = (
    "energy_total_J",
    "energy_rel_error",
)

PLANET_DATASET_NAMES = (
    "dx_au", "dy_au", "dz_au",
    "x_au", "y_au", "z_au",
    "vx_kms", "vy_kms", "vz_kms",
)


def resolve_r_star_au_value(r_star_param):
    r_star_val = r_star_param
    if hasattr(r_star_val, "value_in"):
        r_star_val = r_star_val.value_in(units.RSun)
    return r_star_val


def build_static_metadata(params, r_star_val):
    return {
        "M_STAR_MSun": params["M_STAR"],
        "R_STAR_RSun": r_star_val,
        "T_FINAL_DAYS": params["T_FINAL_DAYS"],
        "N_POINTS_PER_TRANSIT": params["N_POINTS_PER_TRANSIT"],
        "COARSE_FACTOR": params["COARSE_FACTOR"],
        "BUFFER_FACTOR": params["BUFFER_FACTOR"],
        "MAX_COARSE_STEP_DAYS": params["MAX_COARSE_STEP_DAYS"],
        "ENERGY_ERROR_THRESHOLD": params["ENERGY_ERROR_THRESHOLD"],
    }


def write_static_metadata(meta_group, params, r_star_val):
    static = build_static_metadata(params, r_star_val)
    for key in METADATA_KEYS:
        meta_group.attrs[key] = static[key]


def write_runtime_metadata(meta_group, n_steps_saved, runtime_seconds, max_energy_error):
    meta_group.attrs["n_steps_saved"] = n_steps_saved
    meta_group.attrs["runtime_seconds"] = runtime_seconds
    meta_group.attrs["max_energy_error"] = max_energy_error
