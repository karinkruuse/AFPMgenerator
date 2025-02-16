import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Fourier coefficients
def compute_fourier_coeffs(B0, alpha, P, num_harmonics):
    """
    Compute Fourier coefficients B_k for a given magnetization profile.

    Parameters:
        B0 (float): Maximum remanence (T)
        alpha (float): Magnet arc angle in radians
        P (int): Number of pole pairs
        num_harmonics (int): Number of harmonics to consider

    Returns:
        B_k (array): Fourier coefficients for each harmonic k
        harmonics (array): Corresponding harmonic orders
    """
    harmonics = np.arange(1, num_harmonics * 2, 2)  # Only odd harmonics
    B_k = (4 * B0 / (np.pi * harmonics)) * np.sin(harmonics * P * alpha / 2)
    return B_k, harmonics

# Function to calculate B-field using Fourier series
def compute_b_field(theta, B_k, harmonics, P):
    """
    Compute the B-field at different angular positions using Fourier series.

    Parameters:
        theta (array): Angular positions (radians)
        B_k (array): Fourier coefficients
        harmonics (array): Harmonic orders
        P (int): Number of pole pairs

    Returns:
        B_field (array): Magnetic flux density at each theta
    """
    B_field = np.zeros_like(theta)
    for k, B_k_value in zip(harmonics, B_k):
        B_field += B_k_value * np.cos(k * P * theta)
    return B_field

# Define system parameters
B0 = 1.2  # Tesla (remanence of magnets)
alpha = np.radians(30)  # Magnet arc angle in radians
P = 6  # Number of pole pairs
num_harmonics = 10  # How many harmonics to sum

# Generate Fourier coefficients
B_k, harmonics = compute_fourier_coeffs(B0, alpha, P, num_harmonics)

# Define angular positions for plotting
theta = np.linspace(-np.pi, np.pi, 1000)

# Compute B-field distribution
B_field = compute_b_field(theta, B_k, harmonics, P)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(np.degrees(theta), B_field, label=f"B-field Distribution ({num_harmonics} harmonics)", linewidth=2)
plt.xlabel("Angular Position (degrees)")
plt.ylabel("Magnetic Flux Density (T)")
plt.title("Axial Flux Generator B-field Distribution (Fourier Approximation)")
plt.axhline(0, color="black", linestyle="--")
plt.legend()
plt.grid()
plt.show()
