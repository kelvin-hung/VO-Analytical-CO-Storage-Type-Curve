# vo_typecurve_core.py
import numpy as np
import pandas as pd
import os

# -------------------- LOAD CALIBRATED PARAMETERS --------------------
PARAM_FILE     = "co2_analytic_typecurve_params_train_only.csv"
FRACTIONS_FILE = "co2_P50_component_fractions.csv"

_params = pd.read_csv(PARAM_FILE).iloc[0]
M1   = float(_params["M1"])
Minf = float(_params["Minf"])
a    = float(_params["a"])
b    = float(_params["b"])

# component fractions (if available)
_has_components = os.path.exists(FRACTIONS_FILE)
if _has_components:
    frac_df = pd.read_csv(FRACTIONS_FILE)
    tD_frac = frac_df["tD"].values
    f_trap  = frac_df["f_trapped"].values
    f_sc    = frac_df["f_supercritical"].values
    f_diss  = frac_df["f_dissolved"].values
else:
    tD_frac = f_trap = f_sc = f_diss = None, None, None, None

# -------------------- MASTER ANALYTIC CURVE -------------------------
def MD_P50_analytic(tD):
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
    tD = np.asarray(tD, float)
    return A * MD_P50_analytic(tD / tstar)

# -------------------- COMPONENT CURVES -------------------------------
def _interp_fraction(x, xgrid, ygrid):
    return np.interp(x, xgrid, ygrid, left=ygrid[0], right=ygrid[-1])

def components_case(tD, A=1.0, tstar=1.0):
    """
    Returns:
        MD_total, MD_trapped, MD_supercritical, MD_dissolved
    If component fractions CSV is missing, trapped/diss/d_sc = 0.
    """
    tD = np.asarray(tD, float)
    MD_tot = MD_case(tD, A=A, tstar=tstar)

    if not _has_components:
        return MD_tot, np.zeros_like(MD_tot), np.zeros_like(MD_tot), np.zeros_like(MD_tot)

    tD_scaled = tD / tstar
    fT = _interp_fraction(tD_scaled, tD_frac, f_trap)
    fS = _interp_fraction(tD_scaled, tD_frac, f_sc)
    fD = _interp_fraction(tD_scaled, tD_frac, f_diss)

    MD_trap = fT * MD_tot
    MD_sc   = fS * MD_tot
    MD_diss = fD * MD_tot
    return MD_tot, MD_trap, MD_sc, MD_diss

# -------------------- CORRELATIONS FOR A, t* ------------------------
beta_A  = np.array([1.3396, 0.3178, -0.0356, -0.0858,
                    0.0019, 0.0504, 0.0367, -0.0192])
beta_ts = np.array([1.3376, 0.3029, -0.0398, -0.0597,
                    -0.0021, 0.0434, 0.0431, -0.0283])

def _features(phi, kh_mD, kvkh, h_m, qinj_1e3Sm3day,
              c_r_1_per_psi, salinity_ppm):
    log_kh   = np.log10(max(kh_mD, 1e-3))
    log_qinj = np.log10(max(qinj_1e3Sm3day, 1e-6))
    log_kvkh = np.log(max(kvkh, 1e-3))
    h100     = h_m / 100.0
    c_r_1e5  = c_r_1_per_psi * 1e5
    sal_1e5  = salinity_ppm / 1e5
    return np.array([1.0, phi, log_kh, log_qinj, log_kvkh,
                     h100, c_r_1e5, sal_1e5])

def estimate_A_tstar(phi, kh_mD, kvkh, h_m, qinj_1e3Sm3day,
                     c_r_1_per_psi, salinity_ppm):
    feat = _features(phi, kh_mD, kvkh, h_m,
                     qinj_1e3Sm3day, c_r_1_per_psi, salinity_ppm)
    A_est     = float(feat @ beta_A)
    tstar_est = float(feat @ beta_ts)
    return A_est, tstar_est
