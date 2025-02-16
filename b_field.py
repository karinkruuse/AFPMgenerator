import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Reads the variables that one gets from onshape and stores them in a dict
# If a value is not a number, it tries to evaluate it as an expression
def load_variables(file_path):
    df = pd.read_csv(file_path)
    variables = {}
    
    for _, row in df.iterrows():
        name = row["Name"].strip()
        value = row["Value"].strip()
        
        try:
            # Try converting to float directly
            variables[name] = float(value)
        except ValueError:
            # If it's not a direct number, attempt evaluation
            try:
                # Replace variable names with their values in the equation
                for var in variables:
                    value = value.replace(f"#{var}", str(variables[var]))
                
                # Safely evaluate the expression with math functions allowed
                variables[name] = eval(value, {"__builtins__": None}, {"math": math, **math.__dict__})
            except Exception:
                variables[name] = f"Cannot evaluate: {value}"
    
    return variables


# Units are mm and deg
file_path = "onshape_variables.csv"
variables = load_variables(file_path)

nr_of_coils = variables["Nr"]
nr_of_phases = 3
nr_of_coils_per_phase = int(nr_of_coils/nr_of_phases)

mu_rm = 1.07
B_remanence = 1.2

unit_conversion = 100

d_stator_outer = variables["OSD"] / unit_conversion
d_stator_inner = variables["ISD"] / unit_conversion
d_stator_mid = (d_stator_outer + d_stator_inner)/2
mag_clearance = variables["mc"] / unit_conversion
mag_side_clearance = variables["msc"]
nr_magnets = variables["mnr"]
nr_magnets = 28
p = int(nr_magnets/2)
print(d_stator_inner, d_stator_outer, d_stator_mid)

# The radial coordinate where the magnets end and start
r_mag_end = (d_stator_outer/2 - mag_clearance)/ unit_conversion 
r_mag_start = (d_stator_mid/2 + mag_clearance) / unit_conversion 
beta = (360/nr_magnets - mag_side_clearance)/180*np.pi # 2*beta is the angular dimension of the magnet
lm = variables["mt"] / unit_conversion
ld = variables["ac"] / unit_conversion

# print(beta, lm, p, ld)
# linear B = B(H) is assumed (ie operating in that region)  
# That what they used in the 2020 modeling paper



harmonics_order = 17
Q = np.arange(-harmonics_order * p, harmonics_order * p + 1, 2*p)

def coefs(r, harm):
    temp = harm*(2*lm + ld)/r
    co = 2*B_remanence/np.pi*p/harm*np.sin(harm*beta)*2*\
    np.sinh(harm*lm/r)*np.cosh(temp/2)/mu_rm/np.sinh(temp)
    return co


# Calculate the B field (air gap)
# the angular coordinate should show the offset from the middle of a magnet
angles = np.linspace(-np.pi/p, np.pi/p, 1000)
r = d_stator_mid
B_PM = []
for angle in angles:
    b = 0
    for harm in Q:
        b += coefs(r, harm)*np.exp(1j*harm*angle)
    B_PM.append(b)


plt.plot(angles, np.real(B_PM))
plt.title()
#plt.legend()
plt.grid()
plt.show()