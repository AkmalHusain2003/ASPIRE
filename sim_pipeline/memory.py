from pathlib import Path
import h5py
import numpy as np
from amuse.units import units
from . import schema

_CHUNK_LEN = 4096
_COMPRESSION = "gzip"

def _create_growable_1d(group, name, dtype=np.float64):
    return group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=(_CHUNK_LEN,),
        compression=_COMPRESSION,
    )

def _append_scalar(dataset, value):
    n = dataset.shape[0]
    dataset.resize((n + 1,))
    dataset[n] = value

class StreamingHDF5Writer:

    def __init__(self, output_path, params, star, planet_names, planets):
        self.output_path = Path(output_path)
        self.tmp_path = Path(str(output_path) + ".tmp")
        self.planet_names = list(planet_names)
        self.n_steps_written = 0

        self._file = h5py.File(self.tmp_path, "w")
        f = self._file

        r_star_val = schema.resolve_r_star_au_value(params["R_STAR"])

        meta = f.create_group("metadata")
        schema.write_static_metadata(meta, params, r_star_val)
        schema.write_runtime_metadata(
            meta, n_steps_saved=0, runtime_seconds=-1.0, max_energy_error=-1.0
        )

        self._ds_time = _create_growable_1d(f, "time_days")

        star_grp = f.create_group("star")
        star_grp.attrs["mass_MSun"] = star.mass.value_in(units.MSun)
        star_grp.attrs["radius_RSun"] = star.radius.value_in(units.RSun)
        self._ds_star = {
            ds_name: _create_growable_1d(star_grp, ds_name)
            for ds_name in schema.STAR_DATASET_NAMES
        }

        sys_grp = f.create_group("system")
        sys_grp.attrs["energy_unit"] = "Joule"
        self._ds_energy_total = _create_growable_1d(sys_grp, "energy_total_J")
        self._ds_energy_rel_error = _create_growable_1d(sys_grp, "energy_rel_error")

        planets_grp = f.create_group("planets")
        self._ds_planet = {}
        for i, name in enumerate(self.planet_names):
            pg = planets_grp.create_group(name)
            pg.attrs["mass_MSun"] = planets[i].mass.value_in(units.MSun)
            pg.attrs["radius_REarth"] = planets[i].radius.value_in(units.REarth)
            self._ds_planet[name] = {
                ds_name: _create_growable_1d(pg, ds_name)
                for ds_name in schema.PLANET_DATASET_NAMES
            }

        f.flush()

    def append_step(
        self,
        t_curr_days,
        star_x_au, star_y_au, star_z_au,
        star_vx_kms, star_vy_kms, star_vz_kms,
        planet_dx_au, planet_dy_au, planet_dz_au,
        planet_x_au, planet_y_au, planet_z_au,
        planet_vx_kms, planet_vy_kms, planet_vz_kms,
        energy_total_J, energy_rel_error,
    ):
        _append_scalar(self._ds_time, t_curr_days)

        _append_scalar(self._ds_star["x_au"], star_x_au)
        _append_scalar(self._ds_star["y_au"], star_y_au)
        _append_scalar(self._ds_star["z_au"], star_z_au)
        _append_scalar(self._ds_star["vx_kms"], star_vx_kms)
        _append_scalar(self._ds_star["vy_kms"], star_vy_kms)
        _append_scalar(self._ds_star["vz_kms"], star_vz_kms)

        _append_scalar(self._ds_energy_total, energy_total_J)
        _append_scalar(self._ds_energy_rel_error, energy_rel_error)

        planet_step_values = {
            "dx_au": planet_dx_au, "dy_au": planet_dy_au, "dz_au": planet_dz_au,
            "x_au": planet_x_au, "y_au": planet_y_au, "z_au": planet_z_au,
            "vx_kms": planet_vx_kms, "vy_kms": planet_vy_kms, "vz_kms": planet_vz_kms,
        }
        for i, name in enumerate(self.planet_names):
            ds = self._ds_planet[name]
            for ds_name in schema.PLANET_DATASET_NAMES:
                _append_scalar(ds[ds_name], planet_step_values[ds_name][i])

        self.n_steps_written += 1

        self._file.flush()

    def finalize(self, runtime_seconds, max_energy_error):
        meta = self._file["metadata"]
        schema.write_runtime_metadata(
            meta,
            n_steps_saved=self.n_steps_written,
            runtime_seconds=runtime_seconds,
            max_energy_error=max_energy_error,
        )

        self._file.flush()
        self._file.close()

        self.tmp_path.replace(self.output_path)

    def abort(self):
        try:
            self._file.flush()
        finally:
            self._file.close()
