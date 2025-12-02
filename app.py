import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import time as t
import tg43_core as tg  # TG-43 formulation engine


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TG-43 Ir-192 Dose Calculation Model",
    page_icon="🧮",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Global VCU-styled CSS
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

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #F3F4F6;
            border-right: 1px solid #E5E7EB;
            padding-top: 0 !important;  /* remove default big padding */
        }

        /* Compact sidebar inner padding */
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0.4rem !important;
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
# Sidebar (VCU compact version with logo pulled up)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='margin-top:-2.8rem;'></div>",   # stronger negative margin
    unsafe_allow_html=True,
)

st.sidebar.image("logo.png", width=160)

st.sidebar.markdown(
    "<hr style='border:0; border-top:1px solid #E5E7EB; margin:0.2rem 0 0.7rem 0;'>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("### TG-43 Ir-192 Dose Tool")

st.sidebar.markdown(
    """
    <ul style="margin-left:0.4rem; margin-top:0.2rem; list-style-type:none;">
      <li><span style="color:#FDB913;">■</span> <b>Set source activity</b></li>
      <li><span style="color:#FDB913;">■</span> <b>Define dwell positions</b></li>
      <li><span style="color:#FDB913;">■</span> <b>Compute point dose</b></li>
      <li><span style="color:#FDB913;">■</span> <b>Generate 2D isodose plots</b></li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# st.sidebar.markdown("---")
# st.sidebar.caption("VCU Massey Comprehensive Cancer Center · Medical Physics")


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
This tool provides a fully interactive implementation of the **TG-43(U1)** 
dose-calculation formalism for the **Varian Ir-192 HDR VS2000** source, 
including point-dose evaluation and multi-dwell isodose visualization.
"""
)

# -----------------------------------------------------------------------------
# Sidebar: global source parameters and dwell definition
# -----------------------------------------------------------------------------
st.sidebar.header("Source & Dwell Setup")

# 1) Source strength (Ci) – we override activity_Ci and Sk in your module
activity_Ci = st.sidebar.number_input(
    "Source activity (Ci)",
    min_value=0.01,
    max_value=20.0,
    value=float(tg.activity_Ci) if hasattr(tg, "activity_Ci") else 10.0,
    step=0.1,
)
# Update globals in tg43_core so all dose_rate calls use this activity
tg.activity_Ci = activity_Ci
tg.Sk = tg.Sk_per_mCi * tg.activity_Ci * 1000.0  # U

st.sidebar.markdown(
    f"**Air-kerma strength** Sk ≈ {tg.Sk:,.0f} U  "
    f" (using Sk_per_mCi = {tg.Sk_per_mCi:.2f} U/mCi)"
)

# # 2) Number of dwells
# n_dwells = st.sidebar.number_input(
#     "Number of dwell positions",
#     min_value=1,
#     max_value=30,
#     value=3,
#     step=1,
# )

# st.sidebar.markdown("### Dwell positions & times")
# st.sidebar.markdown("Define **x, y, z (cm)**, dwell time **t (s)**, and orientation angles.")

# 2) Number of dwells (slider instead of number_input)
n_dwells = st.sidebar.slider(
    "Number of dwell positions",
    min_value=1,
    max_value=30,
    value=3,
    step=1,
)

st.sidebar.markdown("### Dwell positions & times")
st.sidebar.markdown("Define **x, y, z (cm)**, dwell time **t (s)**, and orientation angles.")

# Default dwell table
default_data = {
    "x_cm": [0.0] * n_dwells,
    "y_cm": [0.0] * n_dwells,
    "z_cm": [0.5 * (i - (n_dwells - 1) / 2) for i in range(n_dwells)],  # centered along z
    "t_s": [30.0] * n_dwells,
    # Orientation: theta from +z (0° = +z), phi in x–y plane
    "theta_deg": [0.0] * n_dwells,
    "phi_deg": [0.0] * n_dwells,
}
dwells_df = pd.DataFrame(default_data)

edited_dwells = st.sidebar.data_editor(
    dwells_df,
    num_rows="dynamic",
    use_container_width=True,
    key="dwell_table",
)

# Extract dwell arrays
dwell_xyz = edited_dwells[["x_cm", "y_cm", "z_cm"]].to_numpy(float)       # (N,3)
dwell_t_sec = edited_dwells["t_s"].to_numpy(float)                        # (N,)

# Build orientation unit vectors using your axis_from_polar()
axes_list = []
for th, ph in zip(edited_dwells["theta_deg"], edited_dwells["phi_deg"]):
    axes_list.append(tg.axis_from_polar(th, ph))
dwell_axis_u = np.vstack(axes_list)                                       # (N,3)


# -----------------------------------------------------------------------------
# Tabs for different calculations
# -----------------------------------------------------------------------------
tab_point, tab_plane = st.tabs(["**Point Dose (multi-dwell)**", "**2D Isodose Plane (multi-dwell)**"])

# -----------------------------------------------------------------------------
# Tab 1: Dose at a single point from many dwells
# -----------------------------------------------------------------------------
with tab_point:
    st.subheader("Dose at a Point from Multiple Dwells")

    col1, col2, col3 = st.columns(3)
    xP = col1.number_input("x (cm)", value=1.0, step=0.1, format="%.2f")
    yP = col2.number_input("y (cm)", value=0.0, step=0.1, format="%.2f")
    zP = col3.number_input("z (cm)", value=0.0, step=0.1, format="%.2f")

    if st.button("Compute Point Dose", key="point_dose_button"):
        # Build points array (M,3) with a single point
        P = np.array([[xP, yP, zP]], float)

        # Use your dose_from_dwells() – returns cGy
        dose_cGy = tg.dose_from_dwells(
            points_xyz=P,
            dwell_xyz=dwell_xyz,
            dwell_t_sec=dwell_t_sec,
            dwell_axis_u=dwell_axis_u,
        )
        dose_value = float(dose_cGy[0])

        # Spinner (needs a message and proper indentation)
        with st.spinner("Computing dose..."):
            t.sleep(0.2)

        st.success(
            f"Dose at ({xP:.2f}, {yP:.2f}, {zP:.2f}) = **{dose_value:.3f} cGy** "
            f"(activity = {activity_Ci:.2f} Ci)"
        )

        with st.expander("Dwell table used"):
            st.dataframe(edited_dwells)

# -----------------------------------------------------------------------------
# Tab 2: 2D isodose plane from many dwells (x–z, y = const)
# -----------------------------------------------------------------------------
with tab_plane:
    st.subheader("**2D Isodose in X–Z Plane from Multiple Dwells**")

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
        help="Grid resolution (NxN). 201 is usually fine."
    )

    if st.button("**Compute Isodose Plane**", key="plane_dose_button"):
        xs = np.linspace(x_min, x_max, int(n_pts))
        zs = np.linspace(z_min, z_max, int(n_pts))

        # Use your dose_plane_from_dwells() – returns xs, zs, D[z,x] in cGy
        xs_grid, zs_grid, Dplane = tg.dose_plane_from_dwells(
            xs, zs, y_slice_cm=y_slice,
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
            cf = ax.contourf(xs_grid, zs_grid, Dplane, levels=levels, norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
            cs = ax.contour(xs_grid, zs_grid, Dplane, levels=levels)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

            ax.set_xlabel("x (cm)")
            ax.set_ylabel("z (cm)")
            ax.set_title(
                f"Multi-dwell dose (cGy) at y = {y_slice:.2f} cm\n"
                f"Activity = {activity_Ci:.2f} Ci"
            )
            ax.set_aspect("equal", "box")

            cbar = fig.colorbar(cf)
            cbar.set_label("Dose (cGy)")

            # Spinner (needs a message and proper indentation)
            with st.spinner("Plot isodose lines..."):
                t.sleep(0.2)

            st.pyplot(fig)

            with st.expander("Dwell table used"):
                st.dataframe(edited_dwells)

# balloons 
st.balloons()

st.markdown(
    """
    <div style="
        margin-top: 3rem;
        background-color: #0a0a0a;
        padding: 10px 0;
        text-align: center;
        font-size: 14px;
        letter-spacing: 0.3px;
        border-radius: 6px;
    ">
        <span style="color:#FFD200; font-weight:600;">
            © 2025 Alexander F. I. Osman — All Rights Reserved.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# ############################## * THE END * ######################

# import streamlit as st
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# import matplotlib.ticker as ticker
# import time as t
# import tg43_core as tg  # TG-43 formulation engine


# # -------------------------------------------------------------------------
# # Page configuration
# # -------------------------------------------------------------------------
# st.set_page_config(
#     page_title="TG-43 Ir-192 Dose Calculation Model",
#     page_icon="🧮",
#     layout="wide",
# )


# # -------------------------------------------------------------------------
# # Global VCU-styled CSS (including sticky sidebar)
# # -------------------------------------------------------------------------
# st.markdown(
#     """
#     <style>
#         /* Top VCU gold bar */
#         .vcu-top-bar {
#             height: 4px;
#             width: 100%;
#             background-color: #FDB913;
#             position: fixed;
#             top: 0;
#             left: 0;
#             z-index: 9999;
#         }

#         /* Reduce padding so content sits closer to top bar */
#         .block-container {
#             padding-top: 2.7rem;
#         }

#         /* Sidebar background + remove default padding */
#         [data-testid="stSidebar"] {
#             background-color: #F3F4F6;
#             border-right: 1px solid #E5E7EB;
#             padding-top: 0 !important;
#         }

#         /* Sticky sidebar with independent scrolling */
#         [data-testid="stSidebar"] > div:first-child {
#             position: sticky;
#             top: 0;
#             height: 100vh;
#             overflow-y: auto;
#             padding-top: 0.4rem !important;
#         }

#         /* Sidebar text styling */
#         [data-testid="stSidebar"] h3 {
#             font-size: 1.05rem;
#             margin-bottom: 0.4rem;
#             color: #111827;
#         }

#         [data-testid="stSidebar"] li {
#             font-size: 0.95rem;
#         }

#         /* Tabs styled with VCU gold */
#         .stTabs [data-baseweb="tab"] {
#             font-size: 0.95rem;
#             font-weight: 600;
#             color: #4B5563;
#         }

#         .stTabs [data-baseweb="tab"][aria-selected="true"] {
#             color: #B45309;
#             border-bottom: 3px solid #FDB913;
#         }

#         /* Buttons / number inputs */
#         .stButton>button {
#             background-color: #FDB913;
#             color: #111827;
#             border-radius: 9999px;
#             border: none;
#             font-weight: 600;
#         }

#         .stButton>button:hover {
#             background-color: #F59E0B;
#         }

#         .stNumberInput>div>div>input {
#             border-radius: 0.5rem;
#         }

#         /* Headings */
#         h1, h2, h3 {
#             color: #111827;
#         }
#     </style>

#     <div class="vcu-top-bar"></div>
#     """,
#     unsafe_allow_html=True,
# )


# # -------------------------------------------------------------------------
# # Sidebar (VCU compact version, raised logo)
# # -------------------------------------------------------------------------
# st.sidebar.markdown(
#     "<div style='margin-top:-2.6rem;'></div>",
#     unsafe_allow_html=True,
# )

# st.sidebar.image("logo.png", width=160)

# st.sidebar.markdown(
#     "<hr style='border:0; border-top:1px solid #E5E7EB; margin:0.2rem 0 0.7rem 0;'>",
#     unsafe_allow_html=True,
# )

# st.sidebar.markdown("### TG-43 Ir-192 Dose Tool")

# # Gold bullets
# st.sidebar.markdown(
#     """
#     <ul style="margin-left:0.4rem; margin-top:0.2rem; list-style-type:none;">
#       <li><span style="color:#FDB913;">■</span> <b>Set source activity</b></li>
#       <li><span style="color:#FDB913;">■</span> <b>Define dwell positions</b></li>
#       <li><span style="color:#FDB913;">■</span> <b>Compute point dose</b></li>
#       <li><span style="color:#FDB913;">■</span> <b>Generate 2D isodose plots</b></li>
#     </ul>
#     """,
#     unsafe_allow_html=True,
# )


# # -------------------------------------------------------------------------
# # Main header
# # -------------------------------------------------------------------------
# st.markdown(
#     """
#     <h1 style="margin-top:0.5rem; margin-bottom:0.35rem; font-size:1.85rem;">
#         TG-43(U1) Dose Calculation for the Varian Ir-192 HDR Brachytherapy Source (VS2000)
#     </h1>
#     <hr style="border:0; border-top:2px solid #FDB913; margin:0 0 0.9rem 0;">
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown(
#     """
#     This tool provides a fully interactive implementation of the **TG-43(U1)** 
#     dose-calculation formalism for the **Varian Ir-192 HDR VS2000** source, 
#     including point-dose evaluation and multi-dwell isodose visualization.
#     """
# )


# # -------------------------------------------------------------------------
# # Sidebar: global source parameters
# # -------------------------------------------------------------------------
# st.sidebar.header("Source & Dwell Setup")

# activity_Ci = st.sidebar.number_input(
#     "Source activity (Ci)",
#     min_value=0.01,
#     max_value=20.0,
#     value=float(tg.activity_Ci) if hasattr(tg, "activity_Ci") else 10.0,
#     step=0.1,
# )
# tg.activity_Ci = activity_Ci
# tg.Sk = tg.Sk_per_mCi * tg.activity_Ci * 1000.0

# st.sidebar.markdown(
#     f"**Air-kerma strength** Sk ≈ {tg.Sk:,.0f} U"
# )


# # Number of dwells
# n_dwells = st.sidebar.slider(
#     "Number of dwell positions",
#     min_value=1,
#     max_value=30,
#     value=3,
#     step=1,
# )

# st.sidebar.markdown("### Dwell positions & times")
# st.sidebar.markdown("Define **x, y, z (cm)**, dwell time **t (s)**, and orientation angles.")

# # Default dwell table
# default_data = {
#     "x_cm": [0.0] * n_dwells,
#     "y_cm": [0.0] * n_dwells,
#     "z_cm": [0.5 * (i - (n_dwells - 1) / 2) for i in range(n_dwells)],
#     "t_s": [30.0] * n_dwells,
#     "theta_deg": [0.0] * n_dwells,
#     "phi_deg": [0.0] * n_dwells,
# }
# dwells_df = pd.DataFrame(default_data)

# edited_dwells = st.sidebar.data_editor(
#     dwells_df,
#     num_rows="dynamic",
#     use_container_width=True,
#     key="dwell_table",
# )

# # Extract dwell arrays
# dwell_xyz = edited_dwells[["x_cm", "y_cm", "z_cm"]].to_numpy(float)
# dwell_t_sec = edited_dwells["t_s"].to_numpy(float)

# axes_list = []
# for th, ph in zip(edited_dwells["theta_deg"], edited_dwells["phi_deg"]):
#     axes_list.append(tg.axis_from_polar(th, ph))
# dwell_axis_u = np.vstack(axes_list)


# # -------------------------------------------------------------------------
# # Tabs
# # -------------------------------------------------------------------------
# tab_point, tab_plane = st.tabs(
#     ["**Point Dose (multi-dwell)**", "**2D Isodose Plane (multi-dwell)**"]
# )


# # -------------------------------------------------------------------------
# # Tab 1 — Point dose
# # -------------------------------------------------------------------------
# with tab_point:
#     st.subheader("**Dose at a Point from Multiple Dwells**")

#     col1, col2, col3 = st.columns(3)
#     xP = col1.number_input("x (cm)", value=1.0, step=0.1)
#     yP = col2.number_input("y (cm)", value=0.0, step=0.1)
#     zP = col3.number_input("z (cm)", value=0.0, step=0.1)

#     if st.button("**Compute Point Dose**"):
#         P = np.array([[xP, yP, zP]], float)

#         dose_cGy = tg.dose_from_dwells(
#             points_xyz=P,
#             dwell_xyz=dwell_xyz,
#             dwell_t_sec=dwell_t_sec,
#             dwell_axis_u=dwell_axis_u,
#         )

#         st.success(
#             f"Dose at ({xP:.2f}, {yP:.2f}, {zP:.2f}) = "
#             f"**{dose_cGy[0]:.3f} cGy** (activity = {activity_Ci:.2f} Ci)"
#         )


# # -------------------------------------------------------------------------
# # Tab 2 — 2D isodose
# # -------------------------------------------------------------------------
# with tab_plane:
#     st.subheader("**2D Isodose in X–Z Plane from Multiple Dwells**")

#     colA, colB, colC = st.columns(3)
#     y_slice = colA.number_input("y-slice (cm)", value=0.0, step=0.1)
#     x_min = colB.number_input("x min (cm)", value=-5.0)
#     x_max = colB.number_input("x max (cm)", value=5.0)
#     z_min = colC.number_input("z min (cm)", value=-5.0)
#     z_max = colC.number_input("z max (cm)", value=5.0)

#     n_pts = st.number_input(
#         "Points per axis", min_value=51, max_value=401, value=201, step=50
#     )

#     if st.button("**Compute Isodose Plane**"):
#         xs = np.linspace(x_min, x_max, int(n_pts))
#         zs = np.linspace(z_min, z_max, int(n_pts))

#         xs_grid, zs_grid, Dplane = tg.dose_plane_from_dwells(
#             xs, zs, y_slice_cm=y_slice,
#             dwell_xyz=dwell_xyz,
#             dwell_t_sec=dwell_t_sec,
#             dwell_axis_u=dwell_axis_u,
#         )

#         finite = np.isfinite(Dplane)
#         if not np.any(finite):
#             st.error("All dose values are non-finite.")
#         else:
#             vmax = np.percentile(Dplane[finite], 99.5)
#             vmin = max(vmax / 1e4, 1e-6)
#             levels = np.geomspace(vmin, vmax, 12)

#             fig, ax = plt.subplots(figsize=(6.4, 5.6))
#             cf = ax.contourf(xs_grid, zs_grid, Dplane, levels=levels,
#                              norm=LogNorm(vmin=levels[0], vmax=levels[-1]))
#             cs = ax.contour(xs_grid, zs_grid, Dplane, levels=levels)
#             ax.clabel(cs, inline=True, fontsize=8)

#             ax.set_xlabel("x (cm)")
#             ax.set_ylabel("z (cm)")
#             ax.set_title(
#                 f"Multi-dwell dose (cGy) at y = {y_slice:.2f} cm\n"
#                 f"Activity = {activity_Ci:.2f} Ci"
#             )
#             ax.set_aspect("equal")

#             st.pyplot(fig)


# # -------------------------------------------------------------------------
# # Footer
# # -------------------------------------------------------------------------
# st.markdown(
#     """
#     <hr style='margin-top:50px; border:0; height:1px; background:#333;'>
#     <div style='background:#000; padding:12px 0; text-align:center; border-radius:6px;'>
#         <span style='color:#f6d300; font-weight:600; font-size:13px;'>
#             © 2025 Alexander F. I. Osman — All Rights Reserved.
#         </span>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )


# # END
