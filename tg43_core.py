"""
TG-43(U1) Dose Calculation Core for Varian VS2000 Ir-192 HDR Source.

Implements:
- Line-source geometry factor G_L(r, θ)
- Radial dose function g_L(r)
- 2D anisotropy function F(r, θ)
- Along–away QA(z, away) look-up
- Full TG-43 dose-rate:
    Ḋ(r, θ) = Sk * Λ * [G_L(r, θ) / G_L(1 cm, 90°)] * g_L(r) * F(r, θ)
- 3D dose-rate in Cartesian coordinates
- Multi-dwell dose accumulation with arbitrary source orientations
"""

# © 2025 Alexander F. I. Osman
# Licensed under the MIT License (see LICENSE).

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# =============================================================================
# Load TG-43 data (VariSource Ir-192 VS2000)
# =============================================================================

NPZ_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "tg43_VariSource_Ir192_full.npz",
)
data = np.load(NPZ_PATH)

# Radial dose function gL(r)
r_grid_g = data["r_grid_g"]
g_vals = data["g_vals"]

# Anisotropy function F(r, θ)
r_grid_F = data["r_grid_F"]
theta_grid_F = data["theta_grid_F"]
F_table = data["F_table"]

# Along–away QA(z, away)
z_grid = data["z_grid"]
away_grid = data["away_grid"]
QA_table = data["QA"]

# TG-43 constants (VariSource Ir-192)
Sk_per_mCi = 4.03          # cGy·cm²/h·mCi⁻¹ (U/mCi)
Lambda = 1.100              # cGy/h/U
activity_Ci = 10.0          # default activity (Ci)
Sk = Sk_per_mCi * activity_Ci * 1000.0  # U
L = 0.5                     # effective active length (cm)


# =============================================================================
# Geometry factor G_L(r, θ)
# =============================================================================

def beta_subtended(r_cm, theta_deg, L_cm):
    """
    Subtended angle β for a line source of length L_cm (cm).

    β = atan((z + L/2) / ρ) - atan((z - L/2) / ρ)
    where:
        z   = r cos θ
        ρ   = r sin θ
        r   in cm
        θ   in degrees
    """
    r = np.asarray(r_cm, float)
    theta_rad = np.deg2rad(np.asarray(theta_deg, float))
    L_cm = float(L_cm)

    z = r * np.cos(theta_rad)
    rho = r * np.sin(theta_rad)
    rho = np.where(np.abs(rho) < 1e-12, 1e-12, rho)  # avoid 0 division

    return np.arctan((z + 0.5 * L_cm) / rho) - np.arctan((z - 0.5 * L_cm) / rho)


def Gl(r_cm, theta_deg, L_cm):
    """
    Line-source geometry factor G_L(r, θ) [cm⁻²].

    - θ in degrees (0 ≤ θ ≤ 180)
    - On-axis (θ=0°): G_L = 1 / (r² - L²/4)
    - Off-axis:       G_L = β / (L r sin θ)

    Inputs:
        r_cm      : scalar or array of radial distances (cm)
        theta_deg : scalar or array of polar angles (deg)
        L_cm      : line length (cm)
    """
    r = np.asarray(r_cm, float)
    theta_rad = np.deg2rad(np.asarray(theta_deg, float))
    L_cm = float(L_cm)

    out = np.empty_like(r, dtype=float)
    on_axis = np.isclose(theta_rad, 0.0)

    # On-axis
    denom = r ** 2 - (L_cm ** 2) / 4.0
    if np.any(on_axis):
        den_on = denom[on_axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            out_on = np.where(np.abs(den_on) < 1e-12, np.inf, 1.0 / den_on)
        out[on_axis] = out_on

    # Off-axis
    not_axis = ~on_axis
    if np.any(not_axis):
        beta = beta_subtended(
            r[not_axis],
            np.rad2deg(theta_rad[not_axis]),
            L_cm,
        )
        s = np.sin(theta_rad[not_axis])
        s = np.where(np.abs(s) < 1e-12, 1e-12, s)
        rn = np.where(r[not_axis] < 1e-3, 1e-3, r[not_axis])
        out[not_axis] = beta / (L_cm * rn * s)

    return out


# =============================================================================
# Radial dose function gL(r)
# =============================================================================

def gL(r_cm):
    """
    Linear interpolation of TG-43 radial dose function gL(r).

    - r_cm : scalar or array of radius (cm)
    - Returns gL(r) using linear interpolation on (r_grid_g, g_vals)
    - Values are clamped to table range.
    """
    r = np.asarray(r_cm, float)
    r_clamped = np.clip(r, r_grid_g[0], r_grid_g[-1])
    return np.interp(r_clamped, r_grid_g, g_vals)


def gL_loglin(r_cm):
    """
    Log-linear interpolation of gL(r) (for smoother tails).

    Interpolates in log–log space of (r, gL(r)).
    """
    r = np.asarray(r_cm, float)
    rc = np.clip(r, r_grid_g[0], r_grid_g[-1])
    lg = np.interp(np.log(rc), np.log(r_grid_g), np.log(g_vals))
    return np.exp(lg)


# =============================================================================
# Anisotropy function F(r, θ)
# =============================================================================

def F_interp(r_cm, theta_deg):
    """
    Bilinear interpolation of anisotropy function F(r, θ).

    Inputs:
        r_cm      : scalar or array (cm)
        theta_deg : scalar or array (deg)
    Returns:
        F(r, θ) with the same shape as the broadcasted inputs.
    """
    r = np.asarray(r_cm, float)
    th = np.asarray(theta_deg, float)

    rc = np.clip(r, r_grid_F[0], r_grid_F[-1])
    tc = np.clip(th, theta_grid_F[0], theta_grid_F[-1])

    rf = rc.ravel()
    tf = tc.ravel()

    ir = np.searchsorted(r_grid_F, rf) - 1
    it = np.searchsorted(theta_grid_F, tf) - 1
    ir = np.clip(ir, 0, len(r_grid_F) - 2)
    it = np.clip(it, 0, len(theta_grid_F) - 2)

    r1, r2 = r_grid_F[ir], r_grid_F[ir + 1]
    t1, t2 = theta_grid_F[it], theta_grid_F[it + 1]

    wr = (rf - r1) / (r2 - r1 + 1e-12)
    wt = (tf - t1) / (t2 - t1 + 1e-12)

    f11 = F_table[it,     ir    ]
    f12 = F_table[it,     ir + 1]
    f21 = F_table[it + 1, ir    ]
    f22 = F_table[it + 1, ir + 1]

    f_top = (1.0 - wr) * f11 + wr * f12
    f_bot = (1.0 - wr) * f21 + wr * f22
    Ff = (1.0 - wt) * f_top + wt * f_bot

    return Ff.reshape(rc.shape)


# =============================================================================
# QA along–away (z, away)
# =============================================================================

def QA_interp(z_cm, away_cm, tol=1e-9):
    """
    Bilinear interpolation in the along–away QA table.

    Inputs:
        z_cm    : scalar or array (cm)
        away_cm : scalar or array (cm)
        tol     : snapping tolerance for matching exact grid nodes

    Returns:
        QA(z, away) with the same shape as the broadcasted inputs.
    """
    z_in = np.asarray(z_cm, float)
    a_in = np.asarray(away_cm, float)

    zg = np.asarray(z_grid, float)
    ag = np.asarray(away_grid, float)
    Q = np.asarray(QA_table, float)

    # Ensure ascending axes
    if zg.size > 1 and zg[1] < zg[0]:
        zg = zg[::-1]
        Q = Q[::-1, :]
    if ag.size > 1 and ag[1] < ag[0]:
        ag = ag[::-1]
        Q = Q[:, ::-1]

    def snap_idx(x, grid):
        idx = np.where(np.isclose(grid, x, atol=tol))[0]
        return int(idx[0]) if idx.size else None

    def lin_safe(x, x1, x2, v1, v2):
        if np.isclose(x, x1, atol=tol):
            return v1
        if np.isclose(x, x2, atol=tol):
            return v2
        w = (x - x1) / (x2 - x1 + 1e-12)
        return (1.0 - w) * v1 + w * v2

    out = np.empty_like(z_in, dtype=float)
    it = np.nditer([z_in, a_in, out],
                   op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for zq, aq, dest in it:
        zq = float(zq)
        aq = float(aq)

        zq = np.clip(zq, zg[0], zg[-1])
        aq = np.clip(aq, ag[0], ag[-1])

        izn = snap_idx(zq, zg)
        ian = snap_idx(aq, ag)

        # Exact node
        if izn is not None and ian is not None:
            dest[...] = Q[izn, ian]
            continue

        # On a z-row: interpolate along away
        if izn is not None:
            r = izn
            ia = np.searchsorted(ag, aq) - 1
            ia = int(np.clip(ia, 0, len(ag) - 2))
            a1, a2 = ag[ia], ag[ia + 1]
            v1, v2 = Q[r, ia], Q[r, ia + 1]
            dest[...] = lin_safe(aq, a1, a2, v1, v2)
            continue

        # On an away-column: interpolate along z
        if ian is not None:
            c = ian
            iz = np.searchsorted(zg, zq) - 1
            iz = int(np.clip(iz, 0, len(zg) - 2))
            z1, z2 = zg[iz], zg[iz + 1]
            v1, v2 = Q[iz, c], Q[iz + 1, c]
            dest[...] = lin_safe(zq, z1, z2, v1, v2)
            continue

        # General bilinear
        iz = np.searchsorted(zg, zq) - 1
        ia = np.searchsorted(ag, aq) - 1
        iz = int(np.clip(iz, 0, len(zg) - 2))
        ia = int(np.clip(ia, 0, len(ag) - 2))

        z1, z2 = zg[iz], zg[iz + 1]
        a1, a2 = ag[ia], ag[ia + 1]
        wz = (zq - z1) / (z2 - z1 + 1e-12)
        wa = (aq - a1) / (a2 - a1 + 1e-12)

        if np.isclose(wa, 0.0, atol=tol):
            wa = 0.0
        if np.isclose(wa, 1.0, atol=tol):
            wa = 1.0
        if np.isclose(wz, 0.0, atol=tol):
            wz = 0.0
        if np.isclose(wz, 1.0, atol=tol):
            wz = 1.0

        f11 = Q[iz,   ia  ]
        f12 = Q[iz,   ia + 1]
        f21 = Q[iz + 1, ia  ]
        f22 = Q[iz + 1, ia + 1]

        top = (1.0 - wa) * f11 + wa * f12
        bot = (1.0 - wa) * f21 + wa * f22
        dest[...] = (1.0 - wz) * top + wz * bot

    return out


# =============================================================================
# TG-43 dose-rate Ḋ(r, θ) and Ḋ(x, y, z)
# =============================================================================

G_ref = float(Gl(1.0, 90.0, L))


def dose_rate(r_cm, theta_deg):
    """
    Full TG-43 line-source dose-rate (cGy/h):

        Ḋ(r, θ) = Sk * Λ * [G_L(r, θ) / G_L(1 cm, 90°)] * gL(r) * F(r, θ)

    Inputs:
        r_cm      : scalar or array (cm)
        theta_deg : scalar or array (deg)
    """
    return (
        Sk
        * Lambda
        * (Gl(r_cm, theta_deg, L) / G_ref)
        * gL(r_cm)
        * F_interp(r_cm, theta_deg)
    )


def dose_rate_3d(x_cm, y_cm, z_cm):
    """
    TG-43 dose-rate (cGy/h) at arbitrary Cartesian points (x, y, z),
    assuming source axis aligned with +z.

    Inputs:
        x_cm, y_cm, z_cm : scalar or arrays of shape (...), in cm
    Returns:
        dose-rate array (cGy/h) of the same broadcasted shape.
    """
    X = np.asarray(x_cm, float)
    Y = np.asarray(y_cm, float)
    Z = np.asarray(z_cm, float)

    rho = np.hypot(X, Y)
    r = np.hypot(rho, Z)
    theta_deg = np.rad2deg(np.arctan2(rho, Z)) % 180.0

    return dose_rate(r, theta_deg)


# =============================================================================
# Single-dwell dose-rate planes (for visualization)
# =============================================================================

def dose_rate_plane_xz(x_cm, z_cm, y_slice=0.0):
    """
    Dose-rate (cGy/h) on an X–Z plane at fixed y = y_slice (cm),
    for a single dwell at the origin with axis aligned to +z.
    """
    X, Z = np.meshgrid(np.asarray(x_cm, float),
                       np.asarray(z_cm, float))
    Y = np.full_like(X, float(y_slice))

    rho = np.hypot(X, Y)
    r = np.hypot(rho, Z)
    theta_deg = np.rad2deg(np.arctan2(rho, Z)) % 180.0
    return dose_rate(r, theta_deg)


def dose_rate_plane_xy(x_cm, y_cm, z_slice=0.0):
    """
    Dose-rate (cGy/h) on an X–Y plane at fixed z = z_slice (cm),
    for a single dwell at the origin with axis aligned to +z.
    """
    X, Y = np.meshgrid(np.asarray(x_cm, float),
                       np.asarray(y_cm, float))
    Z = np.full_like(X, float(z_slice))

    rho = np.hypot(X, Y)
    r = np.hypot(rho, Z)
    theta_deg = np.rad2deg(np.arctan2(rho, Z)) % 180.0
    return dose_rate(r, theta_deg)


def dose_rate_plane_yz(y_cm, z_cm, x_slice=0.0):
    """
    Dose-rate (cGy/h) on a Y–Z plane at fixed x = x_slice (cm),
    for a single dwell at the origin with axis aligned to +z.
    """
    Y, Z = np.meshgrid(np.asarray(y_cm, float),
                       np.asarray(z_cm, float))
    X = np.full_like(Y, float(x_slice))

    rho = np.hypot(X, Y)
    r = np.hypot(rho, Z)
    theta_deg = np.rad2deg(np.arctan2(rho, Z)) % 180.0
    return dose_rate(r, theta_deg)


# =============================================================================
# Plot helpers for isodose visualization (single dwell)
# =============================================================================

def _fmt_cbar(x, pos):
    if x < 1:
        return f"{x:.2e}"
    if x < 1000:
        return f"{x:.0f}"
    return f"{x / 1000:.1f}×10³"


def plot_isodose_plane_xz(xmin=-5, xmax=5,
                          zmin=-5, zmax=5,
                          npts=401, y_slice=0.0):
    """
    Plot TG-43 dose-rate isodose contours (cGy/h) on X–Z plane (y = const).
    """
    xs = np.linspace(xmin, xmax, npts)
    zs = np.linspace(zmin, zmax, npts)
    DR = dose_rate_plane_xz(xs, zs, y_slice=y_slice)

    finite = np.isfinite(DR)
    vmax = np.percentile(DR[finite], 99.5)
    vmin = max(vmax / 1e4, 1e-6)
    levels = np.geomspace(vmin, vmax, 12)

    plt.figure(figsize=(7, 6))
    cf = plt.contourf(xs, zs, DR, levels=levels, cmap="jet",
                      norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    cs = plt.contour(xs, zs, DR, levels=levels, colors="k",
                     linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=12,
               fmt=lambda x: f"{x:.0f} cGy/h", colors="black")

    plt.xlabel("x (cm)", fontsize=12)
    plt.ylabel("z (cm)", fontsize=12)
    plt.title("Single dwell (10 Ci Ir-192) — TG-43 Ḋ (cGy/h) — XZ plane",
              fontsize=14)
    plt.axis("equal")

    cbar = plt.colorbar(cf)
    cbar.set_label("Dose-rate (cGy/h)", rotation=90, labelpad=15, fontsize=12)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_cbar))

    plt.legend(
        [plt.Line2D([], [], color="k", lw=0.8)],
        ["Isodose lines (cGy/h)"],
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()
    plt.show()

    return xs, zs, DR


def plot_isodose_plane_xy(xmin=-5, xmax=5,
                          ymin=-5, ymax=5,
                          npts=401, z_slice=0.0):
    """
    Plot TG-43 dose-rate isodose contours (cGy/h) on X–Y plane (z = const).
    """
    xs = np.linspace(xmin, xmax, npts)
    ys = np.linspace(ymin, ymax, npts)
    DR = dose_rate_plane_xy(xs, ys, z_slice=z_slice)

    finite = np.isfinite(DR)
    vmax = np.percentile(DR[finite], 99.5)
    vmin = max(vmax / 1e4, 1e-6)
    levels = np.geomspace(vmin, vmax, 12)

    plt.figure(figsize=(7, 6))
    cf = plt.contourf(xs, ys, DR, levels=levels, cmap="jet",
                      norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    cs = plt.contour(xs, ys, DR, levels=levels, colors="k",
                     linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=12,
               fmt=lambda x: f"{x:.0f} cGy/h", colors="black")

    plt.xlabel("x (cm)", fontsize=12)
    plt.ylabel("y (cm)", fontsize=12)
    plt.title(
        f"Single dwell (10 Ci Ir-192) — TG-43 Ḋ (cGy/h) — XY plane @ z={z_slice:g} cm",
        fontsize=14,
    )
    plt.axis("equal")

    cbar = plt.colorbar(cf)
    cbar.set_label("Dose-rate (cGy/h)", rotation=90, labelpad=15, fontsize=12)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_cbar))

    plt.legend(
        [plt.Line2D([], [], color="k", lw=0.8)],
        ["Isodose lines (cGy/h)"],
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()
    plt.show()

    return xs, ys, DR


def plot_isodose_plane_yz(ymin=-5, ymax=5,
                          zmin=-5, zmax=5,
                          npts=401, x_slice=0.0):
    """
    Plot TG-43 dose-rate isodose contours (cGy/h) on Y–Z plane (x = const).
    """
    ys = np.linspace(ymin, ymax, npts)
    zs = np.linspace(zmin, zmax, npts)
    DR = dose_rate_plane_yz(ys, zs, x_slice=x_slice)

    finite = np.isfinite(DR)
    vmax = np.percentile(DR[finite], 99.5)
    vmin = max(vmax / 1e4, 1e-6)
    levels = np.geomspace(vmin, vmax, 12)

    plt.figure(figsize=(7, 6))
    cf = plt.contourf(ys, zs, DR, levels=levels, cmap="jet",
                      norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
    cs = plt.contour(ys, zs, DR, levels=levels, colors="k",
                     linewidths=0.6, alpha=0.8)
    plt.clabel(cs, inline=True, fontsize=12,
               fmt=lambda x: f"{x:.0f} cGy/h", colors="black")

    plt.xlabel("y (cm)", fontsize=12)
    plt.ylabel("z (cm)", fontsize=12)
    plt.title(
        f"Single dwell (10 Ci Ir-192) — TG-43 Ḋ (cGy/h) — YZ plane @ x={x_slice:g} cm",
        fontsize=14,
    )
    plt.axis("equal")

    cbar = plt.colorbar(cf)
    cbar.set_label("Dose-rate (cGy/h)", rotation=90, labelpad=15, fontsize=12)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_cbar))

    plt.legend(
        [plt.Line2D([], [], color="k", lw=0.8)],
        ["Isodose lines (cGy/h)"],
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()
    plt.show()

    return ys, zs, DR


# =============================================================================
# Multi-dwell dose (arbitrary orientation)
# =============================================================================

def unit(x, y, z):
    """
    Return unit vector in the direction (x, y, z).
    """
    v = np.asarray([x, y, z], float)
    n = np.linalg.norm(v)
    return v / (n if n > 0 else 1.0)


def axis_from_polar(theta_deg=0.0, phi_deg=0.0):
    """
    Build a unit vector from spherical angles (physics convention):

    - theta: angle from +z down (0..180 deg)
    - phi  : azimuth in x–y plane from +x toward +y (0..360 deg)
    """
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    ux = np.sin(th) * np.cos(ph)
    uy = np.sin(th) * np.sin(ph)
    uz = np.cos(th)
    return np.array([ux, uy, uz], float)


def dose_from_dwells(points_xyz, dwell_xyz, dwell_t_sec, dwell_axis_u=None):
    """
    Sum TG-43 dose (cGy) at arbitrary points from multiple dwells.

    Inputs
    ------
    points_xyz   : (M, 3) array of target points (cm) in global coords
    dwell_xyz    : (N, 3) array of dwell positions (cm)
    dwell_t_sec  : (N,)   array of dwell times (s)
    dwell_axis_u : (N, 3) array of unit vectors for each dwell's source axis.
                   If None, defaults to +z for all.

    Returns
    -------
    dose_cGy : (M,) array of total dose (cGy) at each point.
    """
    P = np.asarray(points_xyz, float)              # (M, 3)
    D = np.asarray(dwell_xyz, float)               # (N, 3)
    T = np.asarray(dwell_t_sec, float).reshape(-1) # (N,)
    N = D.shape[0]

    if dwell_axis_u is None:
        U = np.tile(np.array([[0.0, 0.0, 1.0]], float), (N, 1))
    else:
        U = np.asarray(dwell_axis_u, float)
        U = (U.T / np.linalg.norm(U, axis=1, keepdims=True).T).T

    M = P.shape[0]
    dose = np.zeros(M, float)

    for i in range(N):
        r_vec = P - D[i]          # (M, 3) vector from dwell i to each point

        # Decompose along axis U[i]
        z_loc = r_vec @ U[i]      # (M,)
        r_par = np.outer(z_loc, U[i])
        r_perp = r_vec - r_par
        rho = np.linalg.norm(r_perp, axis=1)  # (M,)
        r_mag = np.hypot(rho, z_loc)          # (M,)
        theta_d = np.rad2deg(np.arctan2(rho, z_loc)) % 180.0

        Ddot = dose_rate(r_mag, theta_d)      # cGy/h
        dose += Ddot * (T[i] / 3600.0)

    return dose


def dose_plane_from_dwells(x_cm, z_cm, y_slice_cm,
                           dwell_xyz, dwell_t_sec, dwell_axis_u=None):
    """
    Dose (cGy) on an X–Z plane from multiple dwells at y = y_slice_cm.

    Inputs:
        x_cm, z_cm   : 1D arrays defining the plane grid (cm)
        y_slice_cm   : y coordinate of the plane (cm)
        dwell_xyz    : (N, 3) dwell positions (cm)
        dwell_t_sec  : (N,) dwell times (s)
        dwell_axis_u : (N, 3) dwell orientation unit vectors

    Returns:
        xs, zs, dose_plane (2D array of shape (len(z_cm), len(x_cm)))
    """
    xs = np.asarray(x_cm, float)
    zs = np.asarray(z_cm, float)
    X, Z = np.meshgrid(xs, zs)
    Y = np.full_like(X, float(y_slice_cm))

    pts = np.column_stack(
        [X.ravel(), Y.ravel(), Z.ravel()]
    )  # (M, 3), M = len(xs) * len(zs)

    dose_flat = dose_from_dwells(pts, dwell_xyz, dwell_t_sec, dwell_axis_u)
    return xs, zs, dose_flat.reshape(Z.shape)


# =============================================================================
# Optional quick tests when run as a script
# =============================================================================

if __name__ == "__main__":
    print("# ---------------- TG-43 Core Quick Test ----------------")
    print(f"Sk_per_mCi = {Sk_per_mCi:.3f} U/mCi")
    print(f"Sk         = {Sk:.3f} U")
    print(f"Activity   = {activity_Ci:.3f} Ci")
    print(f"Λ          = {Lambda:.3f} cGy/h/U")
    print(f"gL(1 cm)   = {gL(1):.3f}")
    print(f"F(1, 90°)  = {F_interp(1, 90):.3f}")
    print(f"Ḋ(1 cm, 90°) = {dose_rate(1.0, 90):.3f} cGy/h")
    print(f"Ḋ(1 cm, 90°) = {dose_rate(1.0, 90)/3600.0:.3f} cGy/s")
    print(f"A(QA; z=0, away=2) = {QA_interp(0.0, 2.0):.3f}")
    print("# --------------------------------------------------------")

    # Example plots (comment out if you don't want them when testing)
    # plot_isodose_plane_xz(y_slice=0.0)
    # plot_isodose_plane_xy(z_slice=0.0)
    # plot_isodose_plane_yz(x_slice=0.0)
