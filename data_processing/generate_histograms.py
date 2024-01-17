import numpy as np
import pandas as pd
import os

def read_xvg(fname):
    """Read columns of data from file fname

    Returns numpy array of data
    """
    skip = 0
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if not line.startswith(('#', '@')):
                skip = i
                break

    return np.genfromtxt(fname, skip_header=skip, usecols=(0, 1))


def apply_offset(df, phi_offset=60, psi_offset=-90, phi_mult=1, psi_mult=1):
    """Apply offsets and multipliers to phi and psi."""
    phi = df[0]
    psi = df[1]

    phi = (((phi + 180) + phi_offset) % 360) - 180
    psi = (((psi + 180) + psi_offset) % 360) - 180

    phi *= phi_mult
    psi *= psi_mult

    return pd.DataFrame({'phi': phi, 'psi': psi})


def calculate_ramachandran_histogram(df, phi_offset=60, psi_offset=-90):
    """Calculate the Ramachandran plot histogram."""
    phi_s, phi_e = -180 + phi_offset, 180 + phi_offset
    psi_s, psi_e = -180 + psi_offset, 180 + psi_offset

    histogram, xedges, yedges = np.histogram2d(df['phi'], df['psi'],
                                               range=[[-180, 180], [-180, 180]],
                                               bins=360)

    histogram_data = {
        "histogram": histogram,
        "phi_edges": xedges,
        "psi_edges": yedges
    }

    return histogram_data


# function needing to be ran once just to generate the histograms in 
def generate_save_histograms():
    # Loop through temperatures and concentrations
    for temp in range(280, 361, 10):
        for conc in range(5, 26, 5):
            pwd = os.path.dirname(os.path.realpath(__file__))
            filename = f"{pwd}/data_simulated/10ns_temp{temp}conc0{conc:02}.xvg"
            output_filename = f"{pwd}/data_processed/histogram_temp{temp}_conc0{conc:02}.npy"

            try:
                df = pd.DataFrame(read_xvg(filename))

                hist = calculate_ramachandran_histogram(df)

                # Save histogram data
                np.save(output_filename, hist)
                print(f"Saved histogram for {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}")


generate_save_histograms()