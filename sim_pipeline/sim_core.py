from amuse.lab import Particles, constants
from amuse.units import units, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements, orbital_elements_from_binary
from amuse_huayno.interface import Huayno
import numpy as np
from typing import Dict
from tqdm import tqdm
import time
from .io import package_results
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CALCULATE THE KEPLERIAN PERIODS
# ---------------------------------------------------------------------------
def kepler_period(a, mass1, mass2):
    return (2 * np.pi * (a**3 / (constants.G * (mass1 + mass2))).sqrt())

# ---------------------------------------------------------------------------
# MAKING THE SYSTEM
# ---------------------------------------------------------------------------
def Make_Exoplanet_System(Star_mass: float, 
                          Star_radius: float, 
                          Star_position: np.ndarray,
                          Star_velocity: np.ndarray,
                          list_of_planets: Dict):
    # Mass of the host star
    mass_of_primary = Star_mass | units.MSun

    # Dict of the list of planets
    planets_data = list_of_planets

    # Initialization
    system = Particles(1)
    star = system[0]
    star.name = "Host_Star"
    star.mass = mass_of_primary
    star.radius = Star_radius | units.RSun

    star.position = Star_position | units.au
    star.velocity = Star_velocity | units.kms

    # Make the loop for input all physical params of planet
    for p_data in planets_data:
        temp_binary = new_binary_from_orbital_elements(
            mass1 = star.mass, 
            mass2 = p_data["mass"], 
            semimajor_axis = p_data["semi_major_axis"] | units.au, 
            eccentricity = p_data["ecc"],
            true_anomaly = p_data["true_anomaly"] | units.deg,
            inclination = p_data["inc"] | units.deg,
            longitude_of_the_ascending_node = p_data["loa"] | units.deg,
            argument_of_periapsis = p_data["aop"] | units.deg,
            G = constants.G
        )

        # Calculate the relative position and relative velocity
        rel_position = temp_binary[1].position - temp_binary[0].position
        rel_velocity = temp_binary[1].velocity - temp_binary[0].velocity

        # Make new particle for every single planets
        new_planet = Particles(1)
        new_planet[0].name = p_data["name"]
        new_planet[0].mass = p_data["mass"]
        new_planet[0].radius = p_data["radius"]
        new_planet[0].position = star.position + rel_position
        new_planet[0].velocity = star.velocity + rel_velocity

        # Merge into one system
        system.add_particles(new_planet)

    # Move the com
    system.move_to_center()
    
    return system

# ---------------------------------------------------------------------------
# MAKING THE SYSTEM FROM PRECOMPUTED CARTESIAN STATE
# ---------------------------------------------------------------------------
def Make_Exoplanet_System_Precomputed(Star_mass: float,
                                      Star_radius,
                                      Star_position: np.ndarray,
                                      Star_velocity: np.ndarray,
                                      list_of_planets: Dict):
    # Mass of the host star
    mass_of_primary = Star_mass | units.MSun

    # Radius of the host star: file may already carry an AMUSE unit (e.g. REarth)
    # or a bare float assumed to be in RSun, so convert explicitly to RSun first
    if hasattr(Star_radius, "value_in"):
        radius_of_primary = Star_radius.value_in(units.RSun) | units.RSun
    else:
        radius_of_primary = Star_radius | units.RSun

    # Dict of the list of planets
    planets_data = list_of_planets

    # Initialization
    system = Particles(1)
    star = system[0]
    star.name = "Host_Star"
    star.mass = mass_of_primary
    star.radius = radius_of_primary

    star.position = Star_position | units.au
    star.velocity = Star_velocity | (units.au / units.yr)

    # Make the loop for input all physical params of planet
    for p_data in planets_data:
        # Make new particle for every single planets
        new_planet = Particles(1)
        new_planet[0].name = p_data["name"]
        new_planet[0].mass = p_data["mass"]
        new_planet[0].radius = p_data["radius"]
        new_planet[0].position = [p_data["x_au"], p_data["y_au"], p_data["z_au"]] | units.au
        new_planet[0].velocity = [p_data["vx_au_yr"], p_data["vy_au_yr"], p_data["vz_au_yr"]] | (units.au / units.yr)

        # Merge into one system
        system.add_particles(new_planet)

    # Move the com
    system.move_to_center()

    return system

# ---------------------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------------------
def run_simulation(params, output_path, precomputed_system=None, is_precomputed_params: bool = False):
    if precomputed_system is not None:
        sample_system = precomputed_system
    elif is_precomputed_params:
        sample_system = Make_Exoplanet_System_Precomputed(
            Star_mass=params["M_STAR"],
            Star_radius=params["R_STAR"],
            Star_position=params["STAR_POSITION"],
            Star_velocity=params["STAR_VELOCITY"],
            list_of_planets=params["planets_list"]
        )
    else:
        sample_system = Make_Exoplanet_System(
            Star_mass=params["M_STAR"],
            Star_radius=params["R_STAR"],
            Star_position=params["STAR_POSITION"],
            Star_velocity=params["STAR_VELOCITY"],
            list_of_planets=params["planets_list"]
        )
    print(sample_system)

    # CONVERTER ============================================================
    converter = nbody_system.nbody_to_si(
        sample_system.mass.sum(),
        sample_system[-1].position.length(),
    )

    # HUAYNO GRAVITY =======================================================
    gravity = Huayno(converter, number_of_workers=params["N_WORKERS"])
    gravity.parameters.timestep_parameter = params["GRAVITY_TIMESTEP_PARAM"]
    gravity.parameters.inttype_parameter = 12
    gravity.particles.add_particles(sample_system)
    channel_from_grav_to_bodies = gravity.particles.new_channel_to(sample_system)

    # EXTRACT THE SYSTEM TO GET THE KEPLERIAN PARAMS =======================
    # AND CALCULATE THE TRANSIT TIME =======================================
    star = sample_system[0]
    planets = sample_system[1:]

    periods = []
    for planet in planets:
        temp_bin = Particles()
        temp_bin.add_particle(star)
        temp_bin.add_particle(planet)
        m1, m2, a, _, _, _, _, _ = orbital_elements_from_binary(temp_bin, G=constants.G)
        periods.append(kepler_period(a, m1, m2))

    v_rel = ((planets.vx - star.vx)**2 + (planets.vy - star.vy)**2).sqrt()
    if (v_rel.value_in(units.au / units.day) < 1e-12).any():
        raise ValueError("Degenerate v_rel detected for one or more planets")
    transit_durations = (2.0 * (star.radius + planets.radius) / v_rel).value_in(units.day)

    # ADAPTIVE TIMESTEP THING ==============================================
    N_POINTS_PER_TRANSIT = params["N_POINTS_PER_TRANSIT"]
    COARSE_FACTOR = params["COARSE_FACTOR"]
    BUFFER_FACTOR = params["BUFFER_FACTOR"]
    MIN_TIMESTEP = params["MIN_TIMESTEP_DAYS"] | units.day
    MAX_COARSE_STEP = params["MAX_COARSE_STEP_DAYS"] | units.day
    ENERGY_ERROR_THRESHOLD = params["ENERGY_ERROR_THRESHOLD"]

    min_timestep_days = MIN_TIMESTEP.value_in(units.day)
    max_coarse_step_days = MAX_COARSE_STEP.value_in(units.day)

    fine_steps_days = np.maximum(transit_durations / N_POINTS_PER_TRANSIT, min_timestep_days)

    periods_days = np.array([p.value_in(units.day) for p in periods])
    coarse_step_days = min(periods_days.min() / COARSE_FACTOR, max_coarse_step_days)
    coarse_step_days = max(coarse_step_days, min_timestep_days)

    buffer_zone_au = (BUFFER_FACTOR * (star.radius + planets.radius)).value_in(units.au)

    # SIMULATION BLOCKS =======================================================
    var_time = []
    var_dx_rows = []
    var_dy_rows = []

    var_star_x = []
    var_star_y = []
    var_star_vx = []
    var_star_vy = []

    var_planet_x_rows = []
    var_planet_y_rows = []
    var_planet_vx_rows = []
    var_planet_vy_rows = []

    var_energy_total_J = []
    var_energy_rel_error = []

    t_curr = 0.0 | units.day
    t_final = params['T_FINAL_DAYS'] | units.day
    t_final_days = t_final.value_in(units.day)

    E0 = (gravity.kinetic_energy + gravity.potential_energy)
    max_energy_error = 0.0
    energy_warning_triggered = False

    start = time.time()
    with tqdm(
        total=t_final_days,
        desc="Simulating",
        unit="day",
        bar_format="{bar:50}| {n:.2f}/{total:.2f} d {postfix}") as pbar:
        while t_curr < t_final:
            t_prev = t_curr

            dx_now = (planets.x - star.x).value_in(units.au)
            dy_now = (planets.y - star.y).value_in(units.au)

            star_zone_mask = (dx_now > 0) & (np.abs(dy_now) <= buffer_zone_au)
            star_zone = bool(star_zone_mask.any())

            if star_zone:
                t_step_days = fine_steps_days[star_zone_mask].min()
            else:
                t_step_days = coarse_step_days

            t_step_days = max(t_step_days, min_timestep_days)
            if not star_zone:
                t_step_days = min(t_step_days, max_coarse_step_days)

            t_curr = t_curr + (t_step_days | units.day)
            gravity.evolve_model(t_curr)
            t_curr = gravity.model_time
            channel_from_grav_to_bodies.copy()

            E_now = (gravity.kinetic_energy + gravity.potential_energy)
            rel_energy_error = abs((E_now - E0) / E0)
            if rel_energy_error > max_energy_error:
                max_energy_error = rel_energy_error

            if rel_energy_error > ENERGY_ERROR_THRESHOLD and not energy_warning_triggered:
                print(f"\n[WARNING] |dE/E0| = {rel_energy_error:.3e} "
                    f"exceeded threshold {ENERGY_ERROR_THRESHOLD:.0e} "
                    f"at t = {t_curr.value_in(units.day):.4f} days. "
                    f"Consider a smaller BUFFER_FACTOR margin or higher N_POINTS_PER_TRANSIT.")
                energy_warning_triggered = True

            dx_new = (planets.x - star.x).value_in(units.au)
            dy_new = (planets.y - star.y).value_in(units.au)

            var_time.append(t_curr.value_in(units.day))
            var_dx_rows.append(dx_new)
            var_dy_rows.append(dy_new)

            var_star_x.append(star.x.value_in(units.au))
            var_star_y.append(star.y.value_in(units.au))
            var_star_vx.append(star.vx.value_in(units.kms))
            var_star_vy.append(star.vy.value_in(units.kms))

            var_planet_x_rows.append(planets.x.value_in(units.au))
            var_planet_y_rows.append(planets.y.value_in(units.au))
            var_planet_vx_rows.append(planets.vx.value_in(units.kms))
            var_planet_vy_rows.append(planets.vy.value_in(units.kms))

            var_energy_total_J.append(E_now.value_in(units.J))
            var_energy_rel_error.append(rel_energy_error)

            delta_days = (t_curr - t_prev).value_in(units.day)
            remaining_days = t_final_days - pbar.n
            pbar.update(min(delta_days, remaining_days))

            pbar.set_postfix({
                "dt": f"{t_step_days:.2e}d",
                "zone": "transit" if star_zone else "coarse",
            })

    gravity.stop()
    end = time.time()

    var_dx = np.array(var_dx_rows).T.tolist()
    var_dy = np.array(var_dy_rows).T.tolist()

    planet_x = np.array(var_planet_x_rows).T.tolist()
    planet_y = np.array(var_planet_y_rows).T.tolist()
    planet_vx = np.array(var_planet_vx_rows).T.tolist()
    planet_vy = np.array(var_planet_vy_rows).T.tolist()

    planet_dx = var_dx
    planet_dy = var_dy

    planet_names = list(planets.name)

    planet_x_by_name = {name: planet_x[i] for i, name in enumerate(planet_names)}
    planet_y_by_name = {name: planet_y[i] for i, name in enumerate(planet_names)}
    planet_vx_by_name = {name: planet_vx[i] for i, name in enumerate(planet_names)}
    planet_vy_by_name = {name: planet_vy[i] for i, name in enumerate(planet_names)}
    planet_dx_by_name = {name: planet_dx[i] for i, name in enumerate(planet_names)}
    planet_dy_by_name = {name: planet_dy[i] for i, name in enumerate(planet_names)}

    print(f"Simulation took: {end - start:.2f} seconds")

    # STORE SIMULATION RESULTS =======================================

    final_results = package_results(
        var_time, star, var_star_x, var_star_y, var_star_vx, var_star_vy,
        planet_names, planets, planet_x_by_name, planet_y_by_name, planet_dx_by_name, planet_dy_by_name,
        planet_vx_by_name, planet_vy_by_name, var_energy_rel_error, var_energy_total_J
        )
    
    if hasattr(max_energy_error, "value_in"):
        max_energy_error = max_energy_error.value_in(units.none)

    final_results["max_energy_error"] = max_energy_error
    final_results["runtime_seconds"] = end - start
    return final_results