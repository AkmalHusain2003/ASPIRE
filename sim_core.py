from amuse.lab import Particles, constants
from amuse.units import units, nbody_system
from amuse.ext.orbital_elements import new_binary_from_orbital_elements, orbital_elements_from_binary
from amuse_huayno.interface import Huayno
import numpy as np
from typing import Dict
from tqdm import tqdm
import time
from .io import open_streaming_writer
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CALCULATE THE KEPLERIAN PERIODS
# ---------------------------------------------------------------------------
def kepler_period(a, mass1, mass2):
    """Compute the Keplerian orbital period from Kepler's third law"""
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

    # Orbital elements are extracted per planet from the instantaneous N-body
    # state (not fixed at t=0 input values) so they correctly reflect secular
    #/short-term perturbations from the other planets in the system.
    periods = []
    inclinations_rad = []
    eccentricities = []
    arg_periapsis_rad = []
    semi_major_axes = []
    for planet in planets:
        temp_bin = Particles()
        temp_bin.add_particle(star)
        temp_bin.add_particle(planet)
        m1, m2, a, ecc, _, inc, _, arg_per = orbital_elements_from_binary(
            temp_bin, G=constants.G
        )
        periods.append(kepler_period(a, m1, m2))
        semi_major_axes.append(a)
        inclinations_rad.append(np.deg2rad(inc))
        ecc_val = ecc.value_in(units.none) if hasattr(ecc, "value_in") else ecc
        eccentricities.append(ecc_val)
        arg_periapsis_rad.append(np.deg2rad(arg_per))

    inclinations_rad = np.array(inclinations_rad)
    eccentricities = np.array(eccentricities)
    arg_periapsis_rad = np.array(arg_periapsis_rad)
    semi_major_axes_au = np.array([a.value_in(units.au) for a in semi_major_axes])

    v_rel_3d = ((planets.vx - star.vx)**2 + (planets.vy - star.vy)**2 + (planets.vz - star.vz)**2).sqrt()
    if (v_rel_3d.value_in(units.au / units.day) < 1e-12).any():
        raise ValueError("Degenerate v_rel detected for one or more planets")

    # --- Impact-parameter-aware transit duration (Winn 2010, Eq. 7 & 14) ---
    # b = (a cos(i) / R_star) * (1 - e^2) / (1 + e sin(omega))
    # Kept general in e and omega (not simplified to e=0) so the estimate
    # remains physically correct if eccentric systems are used later.
    r_star_au = star.radius.value_in(units.au)
    b_impact = np.abs(
        (semi_major_axes_au * np.cos(inclinations_rad) / r_star_au)
        * (1.0 - eccentricities**2) / (1.0 + eccentricities * np.sin(arg_periapsis_rad))
    )

    r_star_plus_planet_au = (star.radius + planets.radius).value_in(units.au)
    # Half-chord length of the planet's sky-plane path across the stellar+
    # planet disk of radius (R_star + R_p), at perpendicular offset b*R_star.
    # Clipped at 0: for b >= (R_star+R_p)/R_star the geometry never transits,
    # so the chord degenerates and the fine-step floor below takes over.
    chord_half_length_au = np.sqrt(
        np.maximum(0.0, r_star_plus_planet_au**2 - (b_impact * r_star_au)**2)
    )

    # Sky-plane (transverse) relative velocity: only the (y, z) components
    # move the planet across the star's projected disk. The line-of-sight
    # component v_x is, by definition, motion toward/away from the observer
    # and does not contribute to sweeping the planet across the stellar
    # disk, so it must be excluded here even though it is correctly
    # included in v_rel_3d above for the degeneracy check. Using the full
    # 3D speed here would silently underestimate T_dur (a small but
    # physically incorrect bias that grows with |v_x|, i.e. away from
    # exact conjunction and away from i = 90 deg).
    v_perp = ((planets.vy - star.vy)**2 + (planets.vz - star.vz)**2).sqrt()
    v_perp_au_day = v_perp.value_in(units.au / units.day)
    # Guard: at exact conjunction v_perp > 0 whenever the orbit is not
    # perfectly degenerate (checked above via v_rel_3d); this floor only
    # protects against transient near-zero crossings of v_perp itself
    # (e.g. an orbit instantaneously moving purely along the line of sight,
    # which is physically possible away from conjunction even though
    # v_rel_3d is nonzero there).
    v_perp_au_day = np.maximum(v_perp_au_day, 1e-12)
    transit_durations = (2.0 * chord_half_length_au) / v_perp_au_day


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

    # STREAMING WRITER ========================================================
    planet_names = list(planets.name)
    writer = open_streaming_writer(output_path, params, star, planet_names, planets)

    t_curr = 0.0 | units.day
    t_final = params['T_FINAL_DAYS'] | units.day
    t_final_days = t_final.value_in(units.day)

    E0 = (gravity.kinetic_energy + gravity.potential_energy)
    max_energy_error = 0.0
    energy_warning_triggered = False

    start = time.time()
    try:
        with tqdm(
            total=t_final_days,
            desc="Simulating",
            unit="day",
            bar_format="{bar:50}| {n:.2f}/{total:.2f} d {postfix}") as pbar:
            while t_curr < t_final:
                t_prev = t_curr

                dx_now = (planets.x - star.x).value_in(units.au)
                dy_now = (planets.y - star.y).value_in(units.au)
                dz_now = (planets.z - star.z).value_in(units.au)

                # Sky-plane projected separation (Winn 2010 convention: y-z is
                # the sky plane). For i != 90 deg the planet's closest approach
                # to the star is generally NOT aligned with the y-axis alone, so
                # using |dy| here (as before) would miss transit-zone entry for
                # inclined orbits. rho = sqrt(dy^2+dz^2) is the general form.
                rho_now = np.sqrt(dy_now**2 + dz_now**2)
                star_zone_mask = (dx_now > 0) & (rho_now <= buffer_zone_au)
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
                dz_new = (planets.z - star.z).value_in(units.au)

                # Direct one-for-one replacement of the sixteen `.append(...)`
                # calls above: each value below is computed identically to
                # before (same source expression, same units, same post-evolve
                # timing -- dx_new/dy_new/dz_new rather than dx_now/dy_now/
                # dz_now, matching the original's use of the post-evolve
                # separation), but is now written straight to its HDF5 dataset
                # and flushed to disk instead of being appended to a Python
                # list that would otherwise persist in RAM for the rest of
                # the run.
                writer.append_step(
                    t_curr_days=t_curr.value_in(units.day),
                    star_x_au=star.x.value_in(units.au),
                    star_y_au=star.y.value_in(units.au),
                    star_z_au=star.z.value_in(units.au),
                    star_vx_kms=star.vx.value_in(units.kms),
                    star_vy_kms=star.vy.value_in(units.kms),
                    star_vz_kms=star.vz.value_in(units.kms),
                    planet_dx_au=dx_new,
                    planet_dy_au=dy_new,
                    planet_dz_au=dz_new,
                    planet_x_au=planets.x.value_in(units.au),
                    planet_y_au=planets.y.value_in(units.au),
                    planet_z_au=planets.z.value_in(units.au),
                    planet_vx_kms=planets.vx.value_in(units.kms),
                    planet_vy_kms=planets.vy.value_in(units.kms),
                    planet_vz_kms=planets.vz.value_in(units.kms),
                    energy_total_J=E_now.value_in(units.J),
                    energy_rel_error=rel_energy_error,
                )

                delta_days = (t_curr - t_prev).value_in(units.day)
                remaining_days = t_final_days - pbar.n
                pbar.update(min(delta_days, remaining_days))

                pbar.set_postfix({
                    "dt": f"{t_step_days:.2e}d",
                    "zone": "transit" if star_zone else "coarse",
                })
    except Exception:
        writer.abort()
        raise
    finally:
        gravity.stop()
    end = time.time()

    print(f"Simulation took: {end - start:.2f} seconds")

    # STORE SIMULATION RESULTS =======================================
    if hasattr(max_energy_error, "value_in"):
        max_energy_error = max_energy_error.value_in(units.none)

    writer.finalize(runtime_seconds=(end - start), max_energy_error=max_energy_error)

    return {
        "output_path": str(output_path),
        "n_steps_saved": writer.n_steps_written,
        "runtime_seconds": end - start,
        "max_energy_error": max_energy_error,
    }