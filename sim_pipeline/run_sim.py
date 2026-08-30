import argparse
from pathlib import Path
from .io import load_input_file, load_precomputed_input_file
from .sim_core import run_simulation 

def main():
    parser = argparse.ArgumentParser(
        description="N-body transit simulation from a .txt input file"
    )
    parser.add_argument("input_file", help="Path to the system config .txt file (Python/AMUSE format)")
    parser.add_argument("-o", "--output", default=None, help="Output HDF5 file path")
    parser.add_argument("--precomputed", action="store_true",
                         help="Treat input_file as a precomputed cartesian state (x_au, y_au, vx_au_yr, vy_au_yr) instead of orbital elements")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    print(f"Reading config from: {input_path}")
    if args.precomputed:
        params = load_precomputed_input_file(input_path)
    else:
        params = load_input_file(input_path)

    if args.output is not None:
        output_path = Path(args.output)
    elif params["Output_file_name"] is not None:
        output_dir = Path(params["Output_Dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{params['Output_file_name']}.h5"
    else:
        output_path = input_path.with_suffix(".h5")

    print(f"Running simulation -> output: {output_path}")
    run_info = run_simulation(params, output_path, is_precomputed_params=args.precomputed)

    print(
        f"\nDone. {run_info['n_steps_saved']} steps saved to: {run_info['output_path']} "
        f"(runtime: {run_info['runtime_seconds']:.2f} s, "
        f"max |dE/E0|: {run_info['max_energy_error']:.3e})"
    )

if __name__ == "__main__":
    main()
