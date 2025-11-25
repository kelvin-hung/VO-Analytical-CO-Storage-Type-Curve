import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from vo_typecurve_core import (
    MD_case,
    components_case,
    estimate_A_tstar,
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="VO CO₂ Storage Type Curve",
                   layout="wide")

st.title("VO Analytical CO₂ Storage Type Curve")
st.markdown(
    "Field-ready tool based on the analytical type-curve framework "
    "and correlations derived from 100 CMG realizations."
)

# ----------------- INPUT SIDEBAR -----------------
st.sidebar.header("Reservoir & operation inputs")

phi   = st.sidebar.slider("Porosity ϕ [-]", 0.05, 0.35, 0.20, 0.01)
kh    = st.sidebar.slider("kh [mD]", 1.0, 1000.0, 150.0, 1.0)
kvkh  = st.sidebar.slider("kv/kh [-]", 0.05, 1.0, 0.5, 0.01)
h_m   = st.sidebar.slider("Net thickness h [m]", 10.0, 300.0, 120.0, 5.0)
qinj  = st.sidebar.slider("q_inj [10³ Sm³/d]", 25.0, 1000.0, 600.0, 25.0)
c_r   = st.sidebar.slider("Rock compressibility [1/psi]",
                          3e-6, 1.5e-5, 9e-6, 1e-6)
sal   = st.sidebar.slider("Salinity [ppm]",
                          3e4, 1.5e5, 1.0e5, 1e4)

t_inj_end = st.sidebar.number_input("Injection duration [years]",
                                    min_value=1.0, max_value=100.0,
                                    value=20.0, step=1.0)
M_inj_Mt  = st.sidebar.number_input("Total injected CO₂ [Mt]",
                                    min_value=1.0, max_value=500.0,
                                    value=50.0, step=1.0)

# ----------------- ESTIMATE A, t* -----------------
A_est, tstar_est = estimate_A_tstar(
    phi, kh, kvkh, h_m, qinj, c_r, sal
)

st.sidebar.markdown("### Estimated scaling parameters")
st.sidebar.write(f"**A ≈ {A_est:.3f}**")
st.sidebar.write(f"**t* ≈ {tstar_est:.3f}**")

# Allow manual override
override = st.sidebar.checkbox("Override A and t*", value=False)
if override:
    A_use = st.sidebar.number_input("A (manual)", 0.1, 3.0, float(A_est), 0.05)
    tstar_use = st.sidebar.number_input("t* (manual)", 0.2, 3.0, float(tstar_est), 0.05)
else:
    A_use, tstar_use = A_est, tstar_est

# ----------------- COMPUTE CURVES -----------------
TMAX_D = 5.0
NPTS   = 400
tD = np.linspace(0.0, TMAX_D, NPTS)

# Dimensionless curves
MD_tot, MD_trap, MD_sc, MD_diss = components_case(tD, A=A_use, tstar=tstar_use)

# Physical time in years
t_years = np.linspace(0.0, 100.0, NPTS)
tD_years = t_years / t_inj_end
MD_tot_y, MD_trap_y, MD_sc_y, MD_diss_y = components_case(tD_years, A=A_use, tstar=tstar_use)
M_tot_Mt  = MD_tot_y  * M_inj_Mt
M_trap_Mt = MD_trap_y * M_inj_Mt
M_sc_Mt   = MD_sc_y   * M_inj_Mt
M_diss_Mt = MD_diss_y * M_inj_Mt

# ----------------- PLOTS -----------------
tab1, tab2 = st.tabs(["Dimensionless curves", "Physical units (Mt vs years)"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(tD, MD_tot,  "k-",  lw=2, label="Total stored")
    ax1.plot(tD, MD_trap, "b-",  lw=2, label="Trapped")
    ax1.plot(tD, MD_sc,   "r--", lw=2, label="Super-critical")
    ax1.plot(tD, MD_diss, "g-.", lw=2, label="Dissolved")
    ax1.axvline(1.0, color="gray", ls="--", lw=1)
    ax1.set_xlabel("tD = t / t_inj_end")
    ax1.set_ylabel("MD component (fraction of injected)")
    ax1.set_title(f"Dimensionless type curves (A={A_use:.2f}, t*={tstar_use:.2f})")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(t_years, M_tot_Mt,  "k-",  lw=2, label="Total stored")
    ax2.plot(t_years, M_trap_Mt, "b-",  lw=2, label="Trapped")
    ax2.plot(t_years, M_sc_Mt,   "r--", lw=2, label="Super-critical")
    ax2.plot(t_years, M_diss_Mt, "g-.", lw=2, label="Dissolved")
    ax2.axvline(t_inj_end, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Time [years]")
    ax2.set_ylabel("CO₂ mass [Mt]")
    ax2.set_title(
        f"CO₂ storage forecast (M_inj = {M_inj_Mt:.1f} Mt)"
    )
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# ----------------- OUTPUT TABLE -----------------
st.subheader("Key outputs at selected times")

for yr in [t_inj_end, 50, 100]:
    if yr > t_years[-1]:
        continue
    idx = np.argmin(np.abs(t_years - yr))
    st.write(
        f"**t = {t_years[idx]:.1f} yr** → "
        f"Total = {M_tot_Mt[idx]:.1f} Mt, "
        f"Trapped = {M_trap_Mt[idx]:.1f} Mt, "
        f"Dissolved = {M_diss_Mt[idx]:.1f} Mt, "
        f"Supercritical = {M_sc_Mt[idx]:.1f} Mt"
    )
# STEP 12 Load P50 component fractions and build component type curves
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# SETTINGS (edit paths if needed)
# -------------------------------------------------
PARAM_FILE     = "co2_analytic_typecurve_params_train_only.csv"
FRACTIONS_FILE = "co2_P50_component_fractions.csv"

TDMAX   = 5.0
N_TDPTS = 400

# -------------------------------------------------
# 1. Load analytic master-curve parameters (M1, Minf, a, b)
# -------------------------------------------------
params = pd.read_csv(PARAM_FILE).iloc[0]
M1   = float(params["M1"])
Minf = float(params["Minf"])
a    = float(params["a"])
b    = float(params["b"])

print("Loaded analytic parameters:")
print(f"  M1   = {M1:.4f}")
print(f"  Minf = {Minf:.4f}")
print(f"  a    = {a:.4f}")
print(f"  b    = {b:.4f}")

# -------------------------------------------------
# 2. Define master analytic curve and scaled case curve
# -------------------------------------------------
def MD_P50_analytic(tD):
    """Analytical P50 dimensionless storage curve."""
    tD = np.asarray(tD, dtype=float)
    MD = np.zeros_like(tD)

    denom = 1.0 - np.exp(-a)
    if abs(denom) < 1e-10:
        denom = 1e-10

    mask1 = tD <= 1.0
    MD[mask1] = M1 * (1.0 - np.exp(-a * tD[mask1])) / denom

    mask2 = tD > 1.0
    MD[mask2] = Minf - (Minf - M1) * np.exp(-b * (tD[mask2] - 1.0))

    return MD


def MD_case(tD, A=1.0, tstar=1.0):
    """Scaled storage curve for a given (A, t*)."""
    return A * MD_P50_analytic(tD / tstar)


# Dense dimensionless time grid
tD_dense = np.linspace(0.0, TDMAX, N_TDPTS)

# -------------------------------------------------
# 3. Load P50 component fractions
# -------------------------------------------------
if not os.path.exists(FRACTIONS_FILE):
    raise FileNotFoundError(
        f"{FRACTIONS_FILE} not found. "
        "Run the script that creates co2_P50_component_fractions.csv first."
    )

frac_df = pd.read_csv(FRACTIONS_FILE)
tD_frac = frac_df["tD"].values
f_trap  = frac_df["f_trapped"].values
f_sc    = frac_df["f_supercritical"].values
f_diss  = frac_df["f_dissolved"].values


def _interp_fraction(x, xgrid, ygrid):
    """Simple 1D interpolation with flat extrapolation."""
    return np.interp(x, xgrid, ygrid, left=ygrid[0], right=ygrid[-1])


def components_case(tD, A=1.0, tstar=1.0):
    """
    Build component curves for a given (A, t*).

    Args:
        tD    : dimensionless time array
        A     : amplitude scaling (final efficiency)
        tstar : time-scale factor

    Returns:
        MD_total, MD_trapped, MD_supercritical, MD_dissolved
    """
    tD = np.asarray(tD, dtype=float)

    # Total curve for this case
    MD_tot = MD_case(tD, A=A, tstar=tstar)

    # Use ensemble P50 fractions evaluated at scaled time tD/t*
    tD_scaled = tD / tstar
    fT = _interp_fraction(tD_scaled, tD_frac, f_trap)
    fS = _interp_fraction(tD_scaled, tD_frac, f_sc)
    fD = _interp_fraction(tD_scaled, tD_frac, f_diss)

    MD_trap = fT * MD_tot
    MD_sc   = fS * MD_tot
    MD_diss = fD * MD_tot

    return MD_tot, MD_trap, MD_sc, MD_diss


# -------------------------------------------------
# 4. Example component type curve in dimensionless form
# -------------------------------------------------
A_ex     = 1.20   # example amplitude
tstar_ex = 1.25   # example time scale

MD_tot_ex, MD_trap_ex, MD_sc_ex, MD_diss_ex = components_case(
    tD_dense, A=A_ex, tstar=tstar_ex
)

plt.figure(figsize=(7, 4))
plt.plot(tD_dense, MD_tot_ex,  "k-",  lw=2, label="Total stored")
plt.plot(tD_dense, MD_trap_ex, "b-",  lw=2, label="Trapped")
plt.plot(tD_dense, MD_sc_ex,   "r--", lw=2, label="Super-critical")
plt.plot(tD_dense, MD_diss_ex, "g-.", lw=2, label="Dissolved")
plt.axvline(1.0, color="gray", ls="--", lw=1)
plt.xlabel("tD = t / t_inj_end")
plt.ylabel("MD component (fraction of injected)")
plt.title(f"Component CO$_2$ storage type curves (A={A_ex:.2f}, t*={tstar_ex:.2f})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fig_typecurve_components_scaled.png", dpi=300)
plt.close()

print("Saved: fig_typecurve_components_scaled.png")

# -------------------------------------------------
# 5. Example in physical units (Mt vs years) for a real project
# -------------------------------------------------
# --- Project input (EDIT these for your scenario) ---
t_inj_end_years = 20.0   # injection duration in years
M_injected_Mt   = 50.0   # total injected CO2 mass in Mt

# time axis for plotting, e.g. 0–100 years
t_years = np.linspace(0.0, 100.0, N_TDPTS)
tD_years = t_years / t_inj_end_years

# compute dimensionless components for this (A_ex, tstar_ex)
MD_tot_y, MD_trap_y, MD_sc_y, MD_diss_y = components_case(
    tD_years, A=A_ex, tstar=tstar_ex
)

# convert to physical mass (Mt)
M_tot_Mt  = MD_tot_y  * M_injected_Mt
M_trap_Mt = MD_trap_y * M_injected_Mt
M_sc_Mt   = MD_sc_y   * M_injected_Mt
M_diss_Mt = MD_diss_y * M_injected_Mt

plt.figure(figsize=(7, 4))
plt.plot(t_years, M_tot_Mt,  "k-",  lw=2, label="Total stored")
plt.plot(t_years, M_trap_Mt, "b-",  lw=2, label="Trapped")
plt.plot(t_years, M_sc_Mt,   "r--", lw=2, label="Super-critical")
plt.plot(t_years, M_diss_Mt, "g-.", lw=2, label="Dissolved")
plt.axvline(t_inj_end_years, color="gray", ls="--", lw=1)
plt.xlabel("Time (years)")
plt.ylabel("CO$_2$ mass (Mt)")
plt.title(
    "CO$_2$ storage components in physical units\n"
    f"(A={A_ex:.2f}, t*={tstar_ex:.2f}, M_inj={M_injected_Mt:.1f} Mt)"
)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fig_typecurve_components_physical_units.png", dpi=300)
plt.close()

print("Saved: fig_typecurve_components_physical_units.png")
