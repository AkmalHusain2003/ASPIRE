import numpy as np
import shapely
from amuse.units import units

def calc_intensity(mu, u1, u2):
    """
    Mandel & Agol Quadratic Limb Darkening
    """
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu)**2

def init_star(r_star, u1, u2, n_annuli=32):
    r_edges = np.linspace(0.0, r_star, n_annuli + 1)
    
    r_mid = np.sqrt((r_edges[:-1]**2 + r_edges[1:]**2) / 2.0)
    mu_mid = np.sqrt(np.clip(1.0 - (r_mid / r_star)**2, 0.0, 1.0))
    
    intensity = calc_intensity(mu_mid, u1, u2)
    
    cum_area = np.pi * r_edges**2
    annulus_area = np.diff(cum_area)
    l_total = np.sum(intensity * annulus_area)
    
    return r_edges, intensity, l_total

def calc_overlap_area(r_edges, r_planet, distance):
    r_edges = np.asarray(r_edges)
    distance = np.asarray(distance)

    area = np.zeros_like(r_edges, dtype=float)
    eps = 1e-12
    
    mask_no = distance >= (r_edges + r_planet - eps)
    mask_full = distance <= (np.abs(r_edges - r_planet) + eps)
    area[mask_full] = np.pi * np.minimum(r_edges[mask_full], r_planet)**2
    
    mask_part = ~(mask_no | mask_full)
    r_part = r_edges[mask_part]
    
    if np.any(mask_part):
        arg1 = (distance**2 + r_part**2 - r_planet**2) / (2 * distance * r_part)
        arg2 = (distance**2 + r_planet**2 - r_part**2) / (2 * distance * r_planet)
        
        arg1 = np.clip(arg1, -1.0, 1.0)
        arg2 = np.clip(arg2, -1.0, 1.0)
        
        star_sector = r_part**2 * np.arccos(arg1)
        planet_sector = r_planet**2 * np.arccos(arg2)
        
        term = r_part**2 - ((r_part**2 - r_planet**2 + distance**2) / (2 * distance))**2
        kite_area = np.sqrt(np.maximum(0, term)) * distance
        
        area[mask_part] = star_sector + planet_sector - kite_area
        
    return area

def calc_flux(r_planet, distance, r_edges, intensity, l_total):
    cum_overlap = calc_overlap_area(r_edges, r_planet, distance)
    annulus_overlap = np.diff(cum_overlap)
    
    l_occulted = np.sum(intensity * annulus_overlap)
    
    return 1.0 - (l_occulted / l_total)

def calc_flux_total(star_center, transiting_planets, r_edges, intensity, l_total, quad_segs=8):
    n_planets = len(transiting_planets)
    sy, sz = star_center
    
    if n_planets == 0:
        return [], 1.0
        
    centers = np.array([p[0] for p in transiting_planets])
    radii = np.array([p[1] for p in transiting_planets])
    
    distances = np.sqrt((centers[:, 0] - sy)**2 + (centers[:, 1] - sz)**2)
    
    individual_fluxes = [
        calc_flux(r, d, r_edges, intensity, l_total) 
        for r, d in zip(radii, distances)
    ]
        
    if n_planets == 1:
        total_flux = individual_fluxes[0]
    else:
        planet_pts = shapely.points(centers)
        planet_circles = shapely.buffer(planet_pts, radii, quad_segs=quad_segs)
        union_planets = shapely.union_all(planet_circles)

        star_pt = shapely.points(star_center)
        disk_polygons = shapely.buffer(star_pt, r_edges[1:], quad_segs=quad_segs)

        intersections = shapely.intersection(disk_polygons, union_planets)
        areas = shapely.area(intersections)

        cum_overlap = np.concatenate(([0.0], areas))
        annulus_overlap = np.diff(cum_overlap)

        l_occulted = np.sum(intensity * annulus_overlap)
        total_flux = 1.0 - (l_occulted / l_total)
        
    return individual_fluxes, total_flux

def prepare_light_curve_inputs(dataframes, planet_names=None):
    r_sun_to_au = (1.0 | units.RSun).value_in(units.au)
    r_earth_to_au = (1.0 | units.REarth).value_in(units.au)

    if planet_names is None:
        planet_names = [name for name in dataframes["attributes"].keys() if name != "star"]

    R_S_au = dataframes["metadata"]["R_STAR_RSun"] * r_sun_to_au

    R_P_au = []
    var_dx = []
    var_dy = []
    var_dz = []
    for name in planet_names:
        radius_REarth = dataframes["attributes"][name]["radius_REarth"]
        R_P_au.append(radius_REarth * r_earth_to_au)

        planet_df = dataframes[name]
        var_dx.append(np.asarray(planet_df["dx_au"]))
        var_dy.append(np.asarray(planet_df["dy_au"]))
        var_dz.append(np.asarray(planet_df["dz_au"]))

    var_time = np.asarray(dataframes["time"]["time_days"])
    n_planets = len(planet_names)

    return R_S_au, R_P_au, n_planets, var_time, var_dx, var_dy, var_dz, planet_names

def make_light_curve(R_S_au, R_P_au, n_planets, var_time, var_dx, var_dy, var_dz,
                      u1_LD=0.5090, u2_LD=0.1925, n_annuli=32, quad_segs=8):
    r_edges, intensity, l_total = init_star(R_S_au, u1_LD, u2_LD, n_annuli=n_annuli)

    var_flux = []
    var_flux_individual = [[] for _ in range(n_planets)]

    for k in range(len(var_time)):
        dxs = [var_dx[i][k] for i in range(n_planets)]
        dys = [var_dy[i][k] for i in range(n_planets)]
        dzs = [var_dz[i][k] for i in range(n_planets)]

        transiting_planets = []
        transiting_indices = []

        for i in range(n_planets):
            rho_i = np.sqrt(dys[i]**2 + dzs[i]**2)
            if dxs[i] > 0 and rho_i < (R_S_au + R_P_au[i]):
                transiting_planets.append(((dys[i], dzs[i]), R_P_au[i]))
                transiting_indices.append(i)

        ind_fluxes, flux_total = calc_flux_total(
            star_center=(0.0, 0.0),
            transiting_planets=transiting_planets,
            r_edges=r_edges,
            intensity=intensity,
            l_total=l_total,
            quad_segs=quad_segs
        )

        var_flux.append(flux_total)

        current_step_fluxes = [1.0] * n_planets
        for idx, f_val in zip(transiting_indices, ind_fluxes):
            current_step_fluxes[idx] = f_val

        for i in range(n_planets):
            var_flux_individual[i].append(current_step_fluxes[i])

    return var_flux, var_flux_individual
