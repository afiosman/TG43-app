# # # ----- MEDP_633: Class Assignment: TG-43(U1) Dose Calculation Model ----------# # #
# © 2025 Alexander F. I. Osman. All rights reserved.

"""
** TG-43(U1) Dose Calculation Algorithm (Varian HDR 192Ir VS2000) **
Aim: develop a model to calculate dose to any random position in space (x,y,z) from any random
array of dwell positions (xi,yi ,zi) and times [sec], for a random strength [Ci] Ir-192 source.
Assume source orientations i are known (so you assign values) for all dwell positions.

** High-level Components:
1. Geometry Factor Gₗ(r, θ):	Fully vectorized, handles on-axis and off-axis cases
2. Radial Dose gₗ(r): Linear + log-linear interpolations
3. Anisotropy F(r, θ):	Bilinear interpolation
4. QA(z, away): Corrected bilinear blend
5. Dose-rate function 𝐷˙(𝑟,𝜃) = 𝑆𝑘 * Λ * 𝐺𝐿(r,θ)/𝐺𝐿(0,90) * gL(r) * 𝐹(𝑟,𝜃)

"""

#######################################################################################################################
# TG-43(U1) Dose Calculation Algorithm #
#######################################################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm


# =========================
#  LOAD THE TG-43 DATA
# =========================

# --- Load TG-43 NPZ data from /data/ for the VS Ir-192 sourcee model
NPZ_PATH = os.path.join(os.path.dirname(__file__), "data", "tg43_VariSource_Ir192_full.npz")
data = np.load(NPZ_PATH)
# print(data.files)  # ['r_grid_g','g_vals','r_grid_F','theta_grid_F','F_table','z_grid','away_grid','QA']

# --- Radial-dose function data gL(r)
r_grid_g = data["r_grid_g"]
g_vals   = data["g_vals"]
# --- Anisotropy function data F(r,θ)
r_grid_F = data["r_grid_F"]
theta_grid_F = data["theta_grid_F"]
F_table  = data["F_table"]
# --- QA along-away data F(z,x,y)
z_grid   = data["z_grid"]
away_grid= data["away_grid"]
QA_table = data["QA"]

# --- TG-43 Constants (VariSource Ir-192 model)
Sk_per_mCi = 4.03           # cGy.cm2/hr.mCi [U/mCi]
Lambda = 1.100              # cGy/hr.U
activity_Ci = 10.0          # Ci
Sk = Sk_per_mCi * activity_Ci * 1000.0  # U
L = 0.5                     # cm (effective active length)


# =========================
#  GEOMETRY FACTORS 𝐺𝐿(r,θ)
# =========================

def beta_subtended(r_cm, theta_deg, L_cm):
    """
    Compute β (subtended angle) for TG-43 line-source geometry.
    β = atan((z+L/2)/ρ) - atan((z-L/2)/ρ).
    where z = r*cosθ, ρ = r*sinθ.
    Inputs: r_cm (float), theta_deg (float), L_cm (float)
    Output: β in radians (float)
    """
    r = np.asarray(r_cm, float)
    theta_rad = np.deg2rad(np.asarray(theta_deg, float))  # degrees → radians
    L_cm = float(L)

    z   = r * np.cos(theta_rad)
    rho = r * np.sin(theta_rad)
    rho = np.where(np.abs(rho) < 1e-12, 1e-12, rho)  # avoid division by zero
    return np.arctan((z + 0.5 * L) / rho) - np.arctan((z - 0.5 * L) / rho)   # radians


def Gl(r_cm, theta_deg, L_cm):
    """
    Line-source geometry factor G_L(r, θ).
    θ in degrees, returns G_L in cm^-2.
    θ=0° → on-axis formula; otherwise β/(L r sinθ).
    """
    r = np.asarray(r_cm, float)
    theta_rad = np.deg2rad(np.asarray(theta_deg, float))
    L_cm = float(L)

    out = np.empty_like(r, dtype=float)
    on_axis = np.isclose(theta_rad, 0.0)  # θ=0° line

    # On-axis (θ=0° mod 180°): G = 1 / (r^2 - L^2/4)
    denom = r ** 2 - (L ** 2) / 4.0
    if np.any(on_axis):
        den_on = denom[on_axis]
        with np.errstate(divide='ignore', invalid='ignore'):
            out_on = np.where(np.abs(den_on) < 1e-12, np.inf, 1.0/den_on)
        out[on_axis] = out_on

    # Off-axis (θ ≠ 0°): G = β / (L r sinθ)
    not_axis = ~on_axis
    if np.any(not_axis):
        beta = beta_subtended(r[not_axis], np.rad2deg(theta_rad[not_axis]), L)
        s = np.sin(theta_rad[not_axis])
        s = np.where(np.abs(s) < 1e-12, 1e-12, s)         # guard sinθ≈0
        rn = np.where(r[not_axis] < 1e-3, 1e-3, r[not_axis])  # guard r→0
        out[not_axis] = beta / (L * rn * s)

    return out


# =========================
#  RADIAL DOSE FUNCTION gL(r)
# =========================

def gL(r_cm):
    """
    Linear interpolation of TG-43 radial dose function gL(r).
    Uses linear interpolation on r_grid_cm vs gL_val
    Clamps r outside the table range
    """
    r = np.asarray(r_cm, float)
    r_clamped = np.clip(r, r_grid_g[0], r_grid_g[-1])
    return np.interp(r_clamped, r_grid_g, g_vals)


def gL_loglin(r_cm):
    """
    Log-space interpolation (for smoother tails)
    """
    r = np.asarray(r_cm, float)
    rc = np.clip(r, r_grid_g[0], r_grid_g[-1])
    lg = np.interp(np.log(rc), np.log(r_grid_g), np.log(g_vals))
    return np.exp(lg)


# =========================
#  ANISOTROPY FUNCTION Ḋ(𝑟,𝜃)
# =========================

def F_interp(r_cm, theta_deg):
    """
    Bilinear interpolation of anisotropy function F(r, θ) with array support.
    r_cm, theta_deg can be scalars or arrays; output matches input shape.
    """
    # clamp to table ranges
    r  = np.asarray(r_cm,   dtype=float)
    th = np.asarray(theta_deg, dtype=float)

    rc = np.clip(r,  r_grid_F[0],     r_grid_F[-1])
    tc = np.clip(th, theta_grid_F[0], theta_grid_F[-1])

    # Flatten
    rf = rc.ravel(); tf = tc.ravel()

    # Find bracketing indices (lower indices for each point)
    ir = np.searchsorted(r_grid_F, rf) - 1
    it = np.searchsorted(theta_grid_F, tf) - 1
    ir = np.clip(ir, 0, len(r_grid_F)-2)
    it = np.clip(it, 0, len(theta_grid_F)-2)

    # Interpolation weights
    r1, r2 = r_grid_F[ir],     r_grid_F[ir+1]
    t1, t2 = theta_grid_F[it], theta_grid_F[it+1]
    wr = (rf - r1) / (r2 - r1 + 1e-12)
    wt = (tf - t1) / (t2 - t1 + 1e-12)

    # Bilinear interpolation (four corners)
    f11 = F_table[it,     ir    ]
    f12 = F_table[it,     ir + 1]
    f21 = F_table[it + 1, ir    ]
    f22 = F_table[it + 1, ir + 1]

    # bilinear blend
    f_top = (1 - wr) * f11 + wr * f12
    f_bot = (1 - wr) * f21 + wr * f22
    Ff = (1 - wt) * f_top + wt * f_bot
    return Ff.reshape(rc.shape)

# =========================
#  QA ALONG-AWAY (z,x,y=0.0)
# =========================

def QA_interp(z_cm, away_cm, tol=1e-9):
    """
    Bilinear interpolation in the along–away QA table.
    - Supports scalars or arrays.
    - Snaps to grid nodes/edges within `tol`.
    """
    z_in = np.asarray(z_cm, float)
    a_in = np.asarray(away_cm, float)

    # Local copies
    zg = np.asarray(z_grid,   float)
    ag = np.asarray(away_grid, float)
    Q  = np.asarray(QA_table,  float)

    # Ensure ascending axes for searchsorted
    if zg.size > 1 and zg[1] < zg[0]:
        zg = zg[::-1]; Q = Q[::-1, :]
    if ag.size > 1 and ag[1] < ag[0]:
        ag = ag[::-1]; Q = Q[:, ::-1]

    def snap_idx(x, grid):
        idx = np.where(np.isclose(grid, x, atol=tol))[0]
        return int(idx[0]) if idx.size else None

    def lin_safe(x, x1, x2, v1, v2):
        if np.isclose(x, x1, atol=tol): return v1
        if np.isclose(x, x2, atol=tol): return v2
        w = (x - x1) / (x2 - x1 + 1e-12)
        return (1 - w) * v1 + w * v2

    out = np.empty_like(z_in, dtype=float)
    it = np.nditer([z_in, a_in, out], op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for zq, aq, dest in it:
        zq = float(zq); aq = float(aq)

        # clamp to grid bounds
        zq = np.clip(zq, zg[0], zg[-1])
        aq = np.clip(aq, ag[0], ag[-1])

        # exact node?
        izn = snap_idx(zq, zg)
        ian = snap_idx(aq, ag)
        if izn is not None and ian is not None:
            dest[...] = Q[izn, ian]
            continue

        # on a z-row? (interpolate along away)
        if izn is not None:
            r = izn
            ia = np.searchsorted(ag, aq) - 1
            ia = int(np.clip(ia, 0, len(ag)-2))
            a1, a2 = ag[ia], ag[ia+1]
            v1, v2 = Q[r, ia], Q[r, ia+1]
            dest[...] = lin_safe(aq, a1, a2, v1, v2)
            continue

        # on an away-column? (interpolate along z)
        if ian is not None:
            c = ian
            iz = np.searchsorted(zg, zq) - 1
            iz = int(np.clip(iz, 0, len(zg)-2))
            z1, z2 = zg[iz], zg[iz+1]
            v1, v2 = Q[iz, c], Q[iz+1, c]
            dest[...] = lin_safe(zq, z1, z2, v1, v2)
            continue

        # general bilinear case
        iz = np.searchsorted(zg, zq) - 1
        ia = np.searchsorted(ag, aq) - 1
        iz = int(np.clip(iz, 0, len(zg)-2))
        ia = int(np.clip(ia, 0, len(ag)-2))

        z1, z2 = zg[iz], zg[iz+1]
        a1, a2 = ag[ia], ag[ia+1]
        wz = (zq - z1) / (z2 - z1 + 1e-12)
        wa = (aq - a1) / (a2 - a1 + 1e-12)

        # snap weights at edges to avoid huge*~0 roundoff
        if np.isclose(wa, 0.0, atol=tol): wa = 0.0
        if np.isclose(wa, 1.0, atol=tol): wa = 1.0
        if np.isclose(wz, 0.0, atol=tol): wz = 0.0
        if np.isclose(wz, 1.0, atol=tol): wz = 1.0

        f11 = Q[iz,   ia  ]   # (z1,a1)
        f12 = Q[iz,   ia+1]   # (z1,a2)
        f21 = Q[iz+1, ia  ]   # (z2,a1)
        f22 = Q[iz+1, ia+1]   # (z2,a2)

        top = (1 - wa) * f11 + wa * f12
        bot = (1 - wa) * f21 + wa * f22
        dest[...] = (1 - wz) * top + wz * bot

    return out


# =========================
#  TG-43 DOSE-RATE Ḋ(𝑟,𝜃)
# =========================

G_ref = float(Gl(1.0, 90.0, L))

def dose_rate(r_cm, theta_deg):
    """
    Full TG-43 line-source dose-rate [cGy/h]:
    Dose-rate(r,θ) = Sk * Lambda * [G(r,θ)/Gref] * gL(r) * F(r,θ)
    r_cm, theta_deg can be scalars or arrays.
    """
    return Sk * Lambda * (Gl(r_cm, theta_deg, L) / G_ref) * gL(r_cm) * F_interp(r_cm, theta_deg)


# =========================
#  TG-43 DOSE-RATE Ḋ(x,y,z)
# =========================

def dose_rate_3d(x_cm, y_cm, z_cm):
    """
    TG-43 dose-rate (cGy/h) for any (x, y, z) coordinates. source axis (aligned with +Z)
    Inputs:
        x_cm, y_cm, z_cm : arrays or scalars in cm
    Returns:
        Dose-rate (cGy/h) array of same shape
    """
    X = np.asarray(x_cm, float)
    Y = np.asarray(y_cm, float)
    Z = np.asarray(z_cm, float)

    # Convert Cartesian → cylindrical relative to source axis (aligned with +Z)
    rho = np.hypot(X, Y)
    r   = np.hypot(rho, Z)
    theta_deg = np.rad2deg(np.arctan2(rho, Z)) % 180.0

    # Full TG-43 dose-rate equation
    return dose_rate(r, theta_deg)


# =========================
#  TG-43 DOSE-RATE PLANE Ḋ(𝑟,𝜃) (single dwell at origin, axis = +z)
# =========================

def dose_rate_plane_xz(x_cm, z_cm, y_slice=0.0):
    """
    Dose-rate (cGy/h) on an X–Z plane (seed axis aligned with +z).
    Dose-rate on an X–Z plane (y = 0), seed at origin, axis = +z
    Returns a 2D array dose_rate[z, x] of dose-rate (cGy/h) over the plane.
    - x_cm, z_cm: 1D arrays of coordinates (cm)
    - y_slice: plane offset in y (cm), default 0
    """
    X, Z = np.meshgrid(np.asarray(x_cm, float), np.asarray(z_cm, float))
    Y = np.full_like(X, float(y_slice))

    # cylindrical w.r.t. source axis (aligned with z)
    rho = np.hypot(X, Y)
    r   = np.hypot(rho, Z)
    theta_deg = (np.rad2deg(np.arctan2(rho, Z)) % 180.0)
    return dose_rate(r, theta_deg)


def dose_rate_plane_xy(x_cm, y_cm, z_slice=0.0):
    """
    Dose-rate (cGy/h) on an X–Y plane (seed axis aligned with +z).
    Dose-rate on an X–Y plane (z = 0), seed at origin, axis = +z
    Returns a 2D array dose_rate[z, Y] of dose-rate (cGy/h) over the plane.
    - x_cm, y_cm: 1D arrays of coordinates (cm)
    - z_slice: plane offset in z (cm), default 0
    """
    X, Y = np.meshgrid(np.asarray(x_cm, float), np.asarray(y_cm, float))
    Z = np.full_like(X, float(z_slice))

    # cylindrical w.r.t. source axis (aligned with z)
    rho = np.hypot(X, Y)
    r   = np.hypot(rho, Z)
    theta_deg = (np.rad2deg(np.arctan2(rho, Z)) % 180.0)
    return dose_rate(r, theta_deg)


def dose_rate_plane_yz(y_cm, z_cm, x_slice=0.0):
    """
    Dose-rate (cGy/h) on an Y–Z plane (seed axis aligned with +z).
    Dose-rate on an Y–Z plane (x = 0), seed at origin, axis = +z
    Returns a 2D array dose_rate[Y, Z] of dose-rate (cGy/h) over the plane.
    - y_cm, z_cm: 1D arrays of coordinates (cm)
    - x_slice: plane offset in x (cm), default 0
    """
    Y, Z = np.meshgrid(np.asarray(y_cm, float), np.asarray(z_cm, float))
    X = np.full_like(Y, float(x_slice))

    # cylindrical w.r.t. source axis (aligned with z)
    rho = np.hypot(X, Y)
    r   = np.hypot(rho, Z)
    theta_deg = (np.rad2deg(np.arctan2(rho, Z)) % 180.0)
    return dose_rate(r, theta_deg)


# =========================
#  PLOT ISODOSE-LINES
# =========================

def plot_isodose_plane_xz(xmin=-5, xmax=5, zmin=-5, zmax=5, npts=401, y_slice=0.0):
    """
    Plot TG-43 isodose contours  along xz plane (dose rate, cGy/h) for a single dwell position
    along z plane.
    Includes labeled colorbar with units and contour level annotations.
    """
    xs = np.linspace(xmin, xmax, npts)
    zs = np.linspace(zmin, zmax, npts)
    DR = dose_rate_plane_xz(xs, zs, y_slice=y_slice)

    # Mask singularities (inf) and choose log-ish spaced levels
    finite = np.isfinite(DR)
    vmax = np.percentile(DR[finite], 99.5)
    vmin = max(vmax/1e4, 1e-6)
    levels = np.geomspace(vmin, vmax, 12)

    # Filled contour (color) and outline contour (lines)
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(xs, zs, DR, levels=levels, cmap='jet',
                      norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    cs = plt.contour(xs, zs, DR, levels=levels, colors='k', linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=12, fmt=lambda x: f"{x:.0f} cGy/h", colors='black')

    # Axis labels and title
    plt.xlabel("x (cm)", fontsize=12)
    plt.ylabel("z (cm)", fontsize=12)
    plt.title("Single dwell (10 Ci Ir-192) — TG-43 Ḋ (cGy/h) — XZ plane", fontsize=14)
    plt.axis("equal")

    cbar = plt.colorbar(cf)
    cbar.set_label("Dose-rate (cGy/h)", rotation=90, labelpad=15, fontsize=12)

    def fmt_dose(x, pos):
        if x < 1: return f"{x:.2e}"
        if x < 1000: return f"{x:.0f}"
        return f"{x/1000:.1f}×10³"
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_dose))

    plt.legend([plt.Line2D([], [], color='k', lw=0.8)], ["Isodose lines (cGy/h)"],
               loc="upper right", frameon=True)
    plt.tight_layout(); plt.show()
    return xs, zs, DR


def plot_isodose_plane_xy(xmin=-5, xmax=5, ymin=-5, ymax=5, npts=401, z_slice=0.0):
    """
    Plot TG-43 isodose contours  along xy plane (dose rate, cGy/h) for a single dwell position
    along z plane.
    Includes labeled colorbar with units and contour level annotations.
    """
    xs = np.linspace(xmin, xmax, npts)
    ys = np.linspace(ymin, ymax, npts)
    DR = dose_rate_plane_xy(xs, ys, z_slice=z_slice)

    # Mask singularities (inf) and choose log-ish spaced levels
    finite = np.isfinite(DR)
    vmax = np.percentile(DR[finite], 99.5)
    vmin = max(vmax/1e4, 1e-6)
    levels = np.geomspace(vmin, vmax, 12)

    # Filled contour (color) and outline contour (lines)
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(xs, ys, DR, levels=levels, cmap='jet',
                      norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    cs = plt.contour(xs, ys, DR, levels=levels, colors='k', linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=12, fmt=lambda x: f"{x:.0f} cGy/h", colors='black')

    # Axis labels and title
    plt.xlabel("x (cm)", fontsize=12)
    plt.ylabel("y (cm)", fontsize=12)
    plt.title(f"Single dwell (10 Ci Ir-192) — TG-43 Ḋ (cGy/h) — XY plane @ z={z_slice:g} cm",
              fontsize=14)
    plt.axis("equal")

    cbar = plt.colorbar(cf)
    cbar.set_label("Dose-rate (cGy/h)", rotation=90, labelpad=15, fontsize=12)

    def fmt_dose(x, pos):
        if x < 1: return f"{x:.2e}"
        if x < 1000: return f"{x:.0f}"
        return f"{x/1000:.1f}×10³"
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_dose))

    plt.legend([plt.Line2D([], [], color='k', lw=0.8)], ["Isodose lines (cGy/h)"],
               loc="upper right", frameon=True)
    plt.tight_layout(); plt.show()
    return xs, ys, DR


def plot_isodose_plane_yz(ymin=-5, ymax=5, zmin=-5, zmax=5, npts=401, x_slice=0.0):
    """
    Plot TG-43 isodose contours  along yz plane (dose rate, cGy/h) for a single dwell position
    along z plane.
    Includes labeled colorbar with units and contour level annotations.
    """
    ys = np.linspace(ymin, ymax, npts)
    zs = np.linspace(zmin, zmax, npts)
    DR = dose_rate_plane_yz(ys, zs, x_slice=x_slice)

    # Mask singularities (inf) and choose log-ish spaced levels
    finite = np.isfinite(DR)
    vmax = np.percentile(DR[finite], 99.5)
    vmin = max(vmax/1e4, 1e-6)
    levels = np.geomspace(vmin, vmax, 12)

    # Filled contour (color) and outline contour (lines)
    plt.figure(figsize=(7, 6))
    cf = plt.contourf(ys, zs, DR, levels=levels, cmap='jet',
                      norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    cs = plt.contour(ys, zs, DR, levels=levels, colors='k', linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=12, fmt=lambda x: f"{x:.0f} cGy/h", colors='black')

    # Axis labels and title
    plt.xlabel("y (cm)", fontsize=12)
    plt.ylabel("z (cm)", fontsize=12)
    plt.title(f"Single dwell (10 Ci Ir-192) — TG-43 Ḋ (cGy/h) — YZ plane @ x={x_slice:g} cm",
              fontsize=14)
    plt.axis("equal")

    cbar = plt.colorbar(cf)
    cbar.set_label("Dose-rate (cGy/h)", rotation=90, labelpad=15, fontsize=12)

    def fmt_dose(x, pos):
        if x < 1: return f"{x:.2e}"
        if x < 1000: return f"{x:.0f}"
        return f"{x/1000:.1f}×10³"
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_dose))

    plt.legend([plt.Line2D([], [], color='k', lw=0.8)], ["Isodose lines (cGy/h)"],
               loc="upper right", frameon=True)
    plt.tight_layout(); plt.show()
    return ys, zs, DR


# =========================
#  EXAMPLE TESTS
# =========================

print("# *------------------------* #")
print(f"Sk_per_mCi = {Sk_per_mCi:.3f}")
print(f"Sk = {Sk:.3f} U")
print(f"Activity = {activity_Ci:.3f} Ci")
print(f"Λ = {Lambda:.3f} cGy/h/U")
print(f"Gl(1.0 cm, 90°, 0.5 cm) = {(Gl(1.0, 90, L)/Gl(1, 90, L)):.3f}")
print(f"gL(1 cm) = {gL(1):.3f}")
print(f"F(1 cm, 90°) = {F_interp(1, 90):.3f}")
# Reference Dose-rate Ḋ(𝑟,𝜃)
print(f"Ḋ(1 cm, 90°) = {dose_rate(1.0, 90):.3f} cGy/h")   # (≈ 44,330 cGy/h)
print(f"Ḋ(1 cm, 90°) = {dose_rate(1.0, 90)/3600.0:.3f} cGy/s")
# On-axis Dose-rate Ḋ(𝑟,𝜃)
print(f"Ḋ(5.0 cm, 0°) = {dose_rate(5.0, 0):.3f} cGy/h")
# Off-axis Dose-rate Ḋ(𝑟,𝜃)
print(f"Ḋ(5 cm, 60°) = {dose_rate(5, 60):.3f} cGy/h")
print(f"A along–away QA(z=0.0, away=2.0) = {QA_interp(0.0, 2.0):.3f}")
# Dose-rate Ḋ(x,y,z) : Single source
print(f"Ḋ(1 cm, 0 cm, 0 cm) = {dose_rate_3d(1.0, 0.0, 0.0):.3f} cGy/h")  # point at (1, 0, 0)
print(f"Ḋ(0 cm, 1 cm, 0 cm) = {dose_rate_3d(0.0, 1.0, 0.0):.3f} cGy/h")  # point at (0, 1, 0)
print(f"Ḋ(0 cm, 0 cm, 1 cm) = {dose_rate_3d(0.0, 0.0, 1.0):.3f} cGy/h")  # on the axis (0, 0, 1)
print(f"Ḋ(2 cm, 2 cm, 2 cm) = {dose_rate_3d(2.0, 2.0, 2.0):.3f} cGy/h")  # point at (2, 2, 2)
print("# *------------------------* #")

# ---- Isodose-line plot (xy, xz, yz)
xs, zs, DR_xz = plot_isodose_plane_xz(y_slice=0.0)   # XZ (Y=0)
xs, ys, DR_xy = plot_isodose_plane_xy(z_slice=0.0)   # XY (Z=0)
ys, zs, DR_yz = plot_isodose_plane_yz(x_slice=0.0)   # YZ (X=0)


# =========================
#  DOSE (X,Y,Z) FROM MULTIPLE DWELL POSITIONS & DWELLS TIMES WITH ARBITRARY ORIENTATIONS
# =========================
# - lets you pass dwells as arrays (x,y,z,t_sec, axis_u),
# - converts each dwell’s orientation into local (r,θ),
# - sums 𝐷˙×𝑡/3600 to return dose (cGy) at any set of points,
# - includes a convenience function to make a dose plane from many dwells.
# It uses the dose_rate(r, theta_deg) function.

# * --- 1) Orientation helpers (unit vectors)

def unit(x, y, z):
    v = np.asarray([x, y, z], float)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)

def axis_from_polar(theta_deg=0.0, phi_deg=0.0):
    """
    Build a unit vector from spherical angles (physics convention):
      - theta: angle from +z down (0..180)
      - phi  : azimuth in x–y plane from +x toward +y (0..360)
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    ux = np.sin(th) * np.cos(ph)
    uy = np.sin(th) * np.sin(ph)
    uz = np.cos(th)
    return np.array([ux, uy, uz], float)


# * --- 2) Core: dose at arbitrary points from many dwells

def dose_from_dwells(points_xyz, dwell_xyz, dwell_t_sec, dwell_axis_u=None):
    """
    Sum TG-43 dose (cGy) at arbitrary 'points_xyz' from multiple dwells.
    Inputs
    -------
    points_xyz : (M,3) array of target points in cm (global coords)
    dwell_xyz  : (N,3) array of dwell positions in cm
    dwell_t_sec: (N,)  dwell times in seconds
    dwell_axis_u: (N,3) array of unit vectors for each dwell's source axis.
                  If None, defaults to +z for all.
    Returns
    -------
    dose_cGy : (M,) array of total dose (cGy) at each target point
    """
    P = np.asarray(points_xyz, float)               # (M,3)
    D = np.asarray(dwell_xyz,  float)               # (N,3)
    T = np.asarray(dwell_t_sec, float).reshape(-1)  # (N,)
    N = D.shape[0]
    if dwell_axis_u is None:
        U = np.tile(np.array([[0.,0.,1.]], float), (N,1))  # +z for all
    else:
        U = np.asarray(dwell_axis_u, float)                  # (N,3)
        # normalize just in case
        U = (U.T / np.linalg.norm(U, axis=1, keepdims=True).T).T

    M = P.shape[0]
    dose = np.zeros(M, float)

    # loop dwells (vectorize over points for each dwell)
    for i in range(N):
        r_vec = P - D[i]               # (M,3) vector from dwell i to each point
        # decompose by orientation axis U[i]
        z_loc   = r_vec @ U[i]         # (M,) projection along axis (+ towards +U)
        r_par   = np.outer(z_loc, U[i])# (M,3) parallel component
        r_perp  = r_vec - r_par         # (M,3) perpendicular component
        rho     = np.linalg.norm(r_perp, axis=1)  # (M,)
        r_mag   = np.hypot(rho, z_loc)           # (M,)
        theta_d = (np.rad2deg(np.arctan2(rho, z_loc)) % 180.0)  # (M,)

        # dose-rate for dwell i, then multiply by time (hr) to get cGy
        Ddot = dose_rate(r_mag, theta_d)         # (M,) cGy/h
        dose += Ddot * (T[i] / 3600.0)
    return dose


# * --- 3) Plane helper for many dwells (x–z slice, y=const)

def dose_plane_from_dwells(x_cm, z_cm, y_slice_cm, dwell_xyz, dwell_t_sec, dwell_axis_u=None):
    """
    Dose (cGy) on an X–Z plane from multiple dwells.
    """
    xs = np.asarray(x_cm, float)
    zs = np.asarray(z_cm, float)
    X, Z = np.meshgrid(xs, zs)                       # (Nz,Nx)
    Y = np.full_like(X, float(y_slice_cm))
    M = X.size
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (M,3)

    dose = dose_from_dwells(pts, dwell_xyz, dwell_t_sec, dwell_axis_u)  # (M,)
    return xs, zs, dose.reshape(Z.shape)


# * --- 4) Example scenario: (3 dwells on a line, +z orientation)
# =========================
#  ASSIGNMENT TEST
# =========================
# Three dwells along +z, spaced 0.5 cm, each oriented +z, with 20/30/25 s
# activity_Ci = 10.0          # Ci
# Sk = Sk_per_mCi * activity_Ci * 1000.0  # U
# dwell_xyz   = np.array([[0, 0, -0.5], [0, 0, 0], [0, 0, 0.5]], float)
# dwell_t_sec = np.array([20, 30, 25], float)
# dwell_axis  = np.tile(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], float), (3, 1))  # +z unit vectors

# Dose at a single point (x=1 cm, y=0, z=0)
# p = np.array([[1.0, 0.0, 0.0]], float)
#dose_at_p = dose_from_dwells(p, dwell_xyz, dwell_t_sec, dwell_axis)
# print("# *---------- *** ASSIGNMENT TEST *** --------------* #")
# print(f"Dose at (1,0,0) cm = {float(dose_at_p[0]):.3f} cGy")
# print("# *-------------------------------------------------* #")


# Dose map on x–z plane (y=0)
# xs = np.linspace(-5, 5, 401)
# zs = np.linspace(-5, 5, 401)
# xs, zs, Dplane = dose_plane_from_dwells(xs, zs, 0.0, dwell_xyz, dwell_t_sec, dwell_axis)

# Plot a contour
# finite = np.isfinite(Dplane)
# vmax = np.percentile(Dplane[finite], 99.5)
# vmin = max(vmax/1e4, 1e-6)
# levels = np.geomspace(vmin, vmax, 12)

# plt.figure(figsize=(6.4,5.6))
# cs = plt.contour(xs, zs, Dplane, levels=levels)
# plt.clabel(cs, inline=True, fontsize=12, fmt="%.0f")
# plt.xlabel("x (cm)", fontsize=12); plt.ylabel("z (cm)", fontsize=12)
# plt.title("Multi-dwell dose (cGy): 3 dwells, +z oriented", fontsize=14)
# plt.axis("equal")
# plt.tight_layout()
# plt.show()


#######################################################################################################################
# **************************************** THE END ****************************************************************** #
#######################################################################################################################
