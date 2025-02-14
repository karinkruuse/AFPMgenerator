import numpy as np
import matplotlib.pyplot as plt

# Constants and Design Parameters
mu_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
Br = 1.2  # Remanence of the PM (T)
p = 4  # Number of pole pairs
r_fixed = 0.1  # Fixed radius for analysis (m)
g = 0.002  # Air gap thickness (m)
h_m = 0.005  # Magnet height (m)
R_rotor = 0.15  # Rotor radius (m)
N_turns = 50  # Number of turns per phase
f = 50  # Electrical frequency in Hz (e.g., at 3000 RPM)
omega = 2 * np.pi * f / p  # Mechanical angular speed (rad/s)

# Increase the number of harmonics for smoother peaks
harmonics_order = 15
harmonics = np.arange(-harmonics_order * p, harmonics_order * p + 1, p)
harmonics = harmonics[harmonics != 0]  # Exclude the zero harmonic

# Angular Positions
theta = np.linspace(0, 2 * np.pi, 500)  # Increased resolution

# Function for B-field based on Fourier approximation
def b_field_fourier(theta, Br, p, harmonics):
    B_axial = np.zeros_like(theta)
    for k in harmonics:
        decay_factor = 1 / (1 + abs(k) / (2 * p))  # Smooth high harmonics
        coefficient = (2 * Br / np.pi) * (p / k) * np.sin(k * np.pi / (2 * p)) * decay_factor
        B_axial += coefficient * np.cos(k * theta)
    return B_axial

# Compute B-field distribution
B_axial = b_field_fourier(theta, Br, p, harmonics)

# Compute Magnetic Flux per coil
coil_span = 2 * np.pi / (2 * p)  # Electrical angle of coil pitch
flux_linkage = (R_rotor - r_fixed) * B_axial * coil_span  # Approximate integral

# Compute EMF using Faraday’s Law
emf = -N_turns * omega * np.gradient(flux_linkage, theta)  # EMF per phase

# Plot the B-field Distribution
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta), B_axial, label="Axial B-field at r={:.3f} m".format(r_fixed))
plt.xlabel("Rotor Position (Degrees)")
plt.ylabel("Magnetic Flux Density B (T)")
plt.title("Axial Flux B-field Distribution in AFPMG")
plt.legend()
plt.grid()
plt.show()

# Plot the Generated EMF
plt.figure(figsize=(8, 6))
plt.plot(np.degrees(theta), emf, label="Induced EMF per phase")
plt.xlabel("Rotor Position (Degrees)")
plt.ylabel("EMF (V)")
plt.title("Generated EMF in AFPMG")
plt.legend()
plt.grid()
plt.show()
