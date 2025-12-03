# app.py  — TG-43 Ir-192 HDR Dose Tool (v0.1.0)

# -----------------------------------------------------------------------------
# Versioning
# -----------------------------------------------------------------------------
APP_VERSION = "0.1.0"  # major.minor.patch
APP_NAME    = "TG-43 Ir-192 Dose Calculation Model"

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import time as t

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, LogNorm

import tg43_core as tg  # TG-43 formulation engine


# -----------------------------------------------------------------------------
# Clinical TPS-style colormap (blue → cyan → green → yellow → red)
# -----------------------------------------------------------------------------
clinical_cmap = LinearSegmentedColormap.from_list(
    "clinical",
    [
        (0.0,  "#00007F"),  # dark blue
        (0.2,  "#0000FF"),  # blue
        (0.4,  "#00FFFF"),  # cyan
        (0.6,  "#00FF00"),  # green
        (0.8,  "#FFFF00"),  # yellow
        (1.0,  "#FF0000"),  # red
    ],
)


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧮",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Global CSS (VCU style + compact sidebar)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Top VCU gold bar */
        .vcu-top-bar {
            height: 4px;
            width: 100%;
            background-color: #FDB913;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 9999;
        }

        /* Reduce top padding so content sits closer to the bar */
        .block-container {
            padding-top: 2.7rem;
        }

        /* Sidebar background + remove default padding */
        [data-testid="stSidebar"] {
            background-color: #F3F4F6;
            border-right: 1px solid #E5E7EB;
            padding-top: 0 !important;
        }

        /* Compact sidebar inner padding */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.4rem !important;
            padding-bottom: 0.4rem !important;
        }

        .sidebar-title {
            margin-top: 0.2rem !important;
            margin-bottom: 0.4rem !important;
        }

        .thin-divider {
            border-top: 1px solid #E5E7EB;
            margin-top: 0.6rem;
            margin-bottom: 0.6rem;
        }

        [data-testid="stSidebar"] h3 {
            font-size: 1.05rem;
            margin-bottom: 0.4rem;
            color: #111827;
        }

        [data-testid="stSidebar"] li {
            font-size: 0.95rem;
        }

        /* Tabs styled with VCU gold */
        .stTabs [data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
            color: #4B5563;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #B45309;
            border-bottom: 3px solid #FDB913;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }

        /* Buttons / number inputs */
        .stButton>button {
            background-color: #FDB913;
            color: #111827;
            border-radius: 9999px;
            border: none;
            font-weight: 600;
        }

        .stButton>button:hover {
            background-color: #F59E0B;
        }

        .stNumberInput>div>div>input {
            border-radius: 0.5rem;
        }

        /* Headings */
        h1, h2, h3 {
            color: #111827;
        }
    </style>

    <div class="vcu-top-bar"></div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: logo, title, version, feature list
# -----------------------------------------------------------------------------
st.sidebar.image("logo.png", width=160)

st.sidebar.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

st.sidebar.markdown(
    "<h3 class='sidebar-title'>TG-43 Ir-192 Dose Tool</h3>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

# Version badge
st.sidebar.markdown(
    f"""
    <div style="font-size:0.85rem; color:#6b7280; margin-bottom:0.3rem;">
        Version
    </div>
    <span style="
        background-color:#d1fae5;
        color:#065f46;
        padding:3px 10px;
        border-radius:8px;
        font-weight:600;
        font-size:0.8rem;">
        v{APP_VERSION}
    </span>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

# Feature list
st.sidebar.markdown(
    """
    <ul style="list-style-type:none; padding-left:0.4rem; margin-top:0.2rem;">
      <li><span style="color:#FDB913;">■</span> Set source activity</li>
      <li><span style="color:#FDB913;">■</span> Define dwell positions</li>
      <li><span style="color:#FDB913;">■</span> Compute point dose</li>
      <li><span style="color:#FDB913;">■</span> Generate 2D isodose plots</li>
    </ul>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main header (title + intro)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <h1 style="margin-top:0.5rem; margin-bottom:0.35rem; font-size:1.85rem;">
        TG-43(U1) Dose Calculation for the Varian Ir-192 HDR Brachytherapy Source (VS2000)
    </h1>
    <hr style="border:0; border-top:2px solid #FDB913; margin:0 0 0.9rem 0;">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
This interactive web app implements the TG-43(U1) formalism for the
Varian Ir-192 HDR VS2000 source. It supports point-dose evaluation
and multi-dwell 2D isodose visualization for educational and research use.
This tool is intended for teaching, prototyping, and independent cross-checks,
not as a clinically commissioned TPS.
"""
)


# -----------------------------------------------------------------------------
# Sidebar: source & dwell setup
# -----------------------------------------------------------------------------
st.sidebar.header("Source & Dwell Setup")

# 1) Source activity (Ci)
activity_Ci = st.sidebar.number_input(
    "Source activity (Ci)",
    min_value=0.01,
    max_value=20.0,
    value=float(tg.activity_Ci) if hasattr(tg, "activity_Ci") else 10.0,
    step=0.1,
)

# Update TG-43 core globals
tg.activity_Ci = activity_Ci
tg.Sk = tg.Sk_per_mCi * tg.activity_Ci * 1000.0  # U

st.sidebar.markdown(
    f"**Air-kerma strength** Sk ≈ {tg.Sk:,.0f} U  "
    f"(Sk_per_mCi = {tg.Sk_per_mCi:.2f} U/mCi)"
)

# 2) Number of dwell positions
n_dwells = st.sidebar.slider(
    "Number of dwell positions",
    min_value=1,
    max_value=30,
    value=3,
    step=1,
)

st.sidebar.markdown("### Dwell positions & times")
st.sidebar.markdown(
    "Define **x, y, z (cm)**, dwell time **t (s)**, and orientation angles."
)

# Default dwell table (centered along z)
default_data = {
    "x_cm": [0.0] * n_dwells,
    "y_cm": [0.0] * n_dwells,
    "z_cm": [0.5 * (i - (n_dwells - 1) / 2) for i in range(n_dwells)],
    "t_s": [30.0] * n_dwells,
    "theta_deg": [0.0] * n_dwells,  # 0° = +z
    "phi_deg": [0.0] * n_dwells,    # x–y plane azimuth
}
dwells_df = pd.DataFrame(default_data)

edited_dwells = st.sidebar.data_editor(
    dwells_df,
    num_rows="dynamic",
    use_container_width=True,
    key="dwell_table",
)

# Extract dwell arrays
dwell_xyz = edited_dwells[["x_cm", "y_cm", "z_cm"]].to_numpy(float)   # (N,3)
dwell_t_sec = edited_dwells["t_s"].to_numpy(float)                    # (N,)

# Orientation unit vectors using tg.axis_from_polar()
axes_list = []
for th, ph in zip(edited_dwells["theta_deg"], edited_dwells["phi_deg"]):
    axes_list.append(tg.axis_from_polar(th, ph))
dwell_axis_u = np.vstack(axes_list)                                   # (N,3)


# -----------------------------------------------------------------------------
# Tabs for different calculations
# -----------------------------------------------------------------------------
tab_point, tab_plane = st.tabs(
    ["**Point Dose (multi-dwell)**", "**2D Isodose Plane (multi-dwell)**"]
)


# -----------------------------------------------------------------------------
# Tab 1: Dose at a single point
# -----------------------------------------------------------------------------
with tab_point:
    st.subheader("Dose at a Point from Multiple Dwells")

    col1, col2, col3 = st.columns(3)
    xP = col1.number_input("x (cm)", value=1.0, step=0.1, format="%.2f")
    yP = col2.number_input("y (cm)", value=0.0, step=0.1, format="%.2f")
    zP = col3.number_input("z (cm)", value=0.0, step=0.1, format="%.2f")

    if st.button("Compute Point Dose", key="point_dose_button"):
        P = np.array([[xP, yP, zP]], float)  # (1,3)

        dose_cGy = tg.dose_from_dwells(
            points_xyz=P,
            dwell_xyz=dwell_xyz,
            dwell_t_sec=dwell_t_sec,
            dwell_axis_u=dwell_axis_u,
        )
        dose_value = float(dose_cGy[0])

        with st.spinner("Computing dose..."):
            t.sleep(0.2)

        st.success(
            f"Dose at ({xP:.2f}, {yP:.2f}, {zP:.2f}) = "
            f"**{dose_value:.3f} cGy** (activity = {activity_Ci:.2f} Ci)"
        )

        with st.expander("Dwell table used"):
            st.dataframe(edited_dwells)


# -----------------------------------------------------------------------------
# Tab 2: 2D isodose plane (x–z, y = const)
# -----------------------------------------------------------------------------
with tab_plane:
    st.subheader("2D Isodose in X–Z Plane from Multiple Dwells")

    colA, colB, colC = st.columns(3)
    y_slice = colA.number_input("y-slice (cm)", value=0.0, step=0.1, format="%.2f")
    x_min = colB.number_input("x min (cm)", value=-5.0, step=0.5)
    x_max = colB.number_input("x max (cm)", value=5.0, step=0.5)
    z_min = colC.number_input("z min (cm)", value=-5.0, step=0.5)
    z_max = colC.number_input("z max (cm)", value=5.0, step=0.5)

    n_pts = st.number_input(
        "Points per axis",
        min_value=51,
        max_value=401,
        value=201,
        step=50,
        help="Grid resolution (N×N). 201 is usually fine.",
    )

    if st.button("Compute Isodose Plane", key="plane_dose_button"):
        xs = np.linspace(x_min, x_max, int(n_pts))
        zs = np.linspace(z_min, z_max, int(n_pts))

        xs_grid, zs_grid, Dplane = tg.dose_plane_from_dwells(
            xs,
            zs,
            y_slice_cm=y_slice,
            dwell_xyz=dwell_xyz,
            dwell_t_sec=dwell_t_sec,
            dwell_axis_u=dwell_axis_u,
        )

        finite = np.isfinite(Dplane)
        if not np.any(finite):
            st.error("All dose values are non-finite. Check dwell definitions.")
        else:
            vmax = np.percentile(Dplane[finite], 99.5)
            vmin = max(vmax / 1e4, 1e-6)
            levels = np.geomspace(vmin, vmax, 12)

            fig, ax = plt.subplots(figsize=(6.4, 5.6))

            # Filled dose map (log scale)
            cf = ax.contourf(
                xs_grid,
                zs_grid,
                Dplane,
                levels=levels,
                cmap=clinical_cmap,
                norm=LogNorm(vmin=levels[0], vmax=levels[-1]),
            )

            # Isodose lines (black)
            cs = ax.contour(
                xs_grid,
                zs_grid,
                Dplane,
                levels=levels,
                colors="black",
                linewidths=0.8,
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

            ax.set_xlabel("x (cm)")
            ax.set_ylabel("z (cm)")
            ax.set_title(
                f"Multi-dwell dose (cGy) at y = {y_slice:.2f} cm\n"
                f"Activity = {activity_Ci:.2f} Ci"
            )
            ax.set_aspect("equal", "box")

            # Colorbar with ticks matching contour levels
            cbar = fig.colorbar(cf, pad=0.03)
            cbar.set_label("Dose (cGy)")
            cbar.set_ticks(levels)
            cbar.ax.set_yticklabels([f"{v:.0f}" for v in levels])

            with st.spinner("Plotting isodose lines..."):
                t.sleep(0.2)

            st.pyplot(fig)

            with st.expander("Dwell table used"):
                st.dataframe(edited_dwells)


# -----------------------------------------------------------------------------
# Celebration & Footer
# -----------------------------------------------------------------------------
st.balloons()

st.markdown(
    """
    <style>
        .footer-box {
            background-color: #000000;
            padding: 14px;
            border-radius: 10px;
            text-align: center;
            color: #F7B718;  /* VCU Gold */
            font-size: 14.5px;
            margin-top: 30px;
            margin-bottom: 10px;
        }
        .footer-link {
            color: #F7B718;
            text-decoration: none;
            font-weight: bold;
        }
        .footer-link:hover {
            color: #ffffff;
        }
    </style>

    <div class="footer-box">
        © 2025 • For questions or suggestions, please contact:<br>
        Alexander F. I. Osman, PhD Student, Department of Radiation Oncology, School of Medicine<br>
        Virginia Commonwealth University<br>
        Email:
        <a class="footer-link" href="mailto:alexanderfadul@yahoo.com">
            alexanderfadul@yahoo.com
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ############################## * THE END * ######################
