import matplotlib
matplotlib.use("Agg")  # CRITICAL: Fixes headless server hang on Streamlit Cloud
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import numpy as np

st.set_page_config(page_title="Column Rebar Detailer", layout="wide")
st.title("Automated Column Detailing (ETABS to ACI 318 Detail)")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Column Geometry & Demand")
p = st.sidebar.number_input("ETABS Reinforcement % (p)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
B = st.sidebar.number_input("Column Width B / Along 3-dir (mm)", min_value=150, max_value=3000, value=1000, step=25)
D = st.sidebar.number_input("Column Depth D / Along 2-dir (mm)", min_value=150, max_value=3000, value=1000, step=25)
stirrup_dia = st.sidebar.selectbox("Stirrup Diameter (mm)", [8, 10, 12, 16], index=2)
cover = st.sidebar.number_input("Clear Cover (mm)", value=40, step=5)

st.sidebar.header("Bar Selection Mode")
mode = st.sidebar.radio("Optimization Mode", ["Auto-Select Best Bar & Layout", "Manual Bar Selection"])

available_dias = [12, 16, 20, 25, 32, 40]
if mode == "Manual Bar Selection":
    dia_choice = st.sidebar.selectbox("Select Bar Diameter (mm)", available_dias, index=4)
    candidate_dias = [dia_choice]
else:
    candidate_dias = available_dias

HARD_MIN_SPACING = 50.0
HARD_MAX_SPACING = 150.0
TARGET_IDEAL_SPACING = 75.0

max_agg_size = 20.0
min_agg_clear_s = (4.0 / 3.0) * max_agg_size  # 26.7 mm

# --- STRUCTURAL DEMAND ---
Ag = B * D
Ast_req = (p * Ag) / 100.0
is_square_column = (B == D)

# --- DETERMINISTIC SOLVER FUNCTION ---
def solve_layout(dia, bundled=False):
    a_bar = (np.pi * (dia ** 2)) / 4.0
    n_bars_needed = int(np.ceil(Ast_req / a_bar))
    if n_bars_needed < 4:
        n_bars_needed = 4

    if bundled:
        n_stations_needed = int(np.ceil(n_bars_needed / 2.0))
        if n_stations_needed < 4:
            n_stations_needed = 4
        de = np.round(np.sqrt(2.0) * dia, 1)
        min_allowable_s = max(de, min_agg_clear_s, HARD_MIN_SPACING)
        min_req_cover = min(de, 50.0)
    else:
        n_stations_needed = n_bars_needed
        de = float(dia)
        min_allowable_s = max(1.5 * dia, min_agg_clear_s, HARD_MIN_SPACING)
        min_req_cover = float(dia)

    span_x = B - 2 * (cover + stirrup_dia) - dia
    span_y = D - 2 * (cover + stirrup_dia) - dia

    max_search = max(25, int(np.ceil(n_stations_needed / 2)) + 6)
    best = None
    min_penalty = float("inf")

    for Nx in range(2, max_search):
        for Ny in range(2, max_search):
            if is_square_column and Nx != Ny:
                continue

            stations = 2 * Nx + 2 * Ny - 4
            if stations >= n_stations_needed:
                if bundled:
                    sx = (span_x - (Nx - 1) * (2 * dia)) / (Nx - 1)
                    sy = (span_y - (Ny - 1) * (2 * dia)) / (Ny - 1)
                else:
                    sx = (span_x - (Nx - 1) * dia) / (Nx - 1)
                    sy = (span_y - (Ny - 1) * dia) / (Ny - 1)

                if sx < min_allowable_s or sy < min_allowable_s:
                    continue
                if sx > HARD_MAX_SPACING or sy > HARD_MAX_SPACING:
                    continue

                total_actual_bars = stations * (2 if bundled else 1)
                Ast_provided = total_actual_bars * a_bar
                p_provided = (Ast_provided / Ag) * 100.0

                excess_steel_ratio = (Ast_provided - Ast_req) / Ast_req
                cost_excess_steel = excess_steel_ratio * 1000.0
                cost_spacing = abs(sx - TARGET_IDEAL_SPACING) + abs(sy - TARGET_IDEAL_SPACING)
                cost_ratio = abs((Nx / Ny) - (B / D)) * 15.0

                total_penalty = cost_excess_steel + cost_spacing + cost_ratio

                if total_penalty < min_penalty:
                    min_penalty = total_penalty
                    best = {
                        "dia": dia,
                        "bundled": bundled,
                        "de": de,
                        "min_allowable_s": min_allowable_s,
                        "min_req_cover": min_req_cover,
                        "Nx": Nx,
                        "Ny": Ny,
                        "stations": stations,
                        "total_bars": total_actual_bars,
                        "Ast_provided": Ast_provided,
                        "p_provided": p_provided,
                        "sx": sx,
                        "sy": sy,
                        "penalty": total_penalty
                    }
    return best

# --- REINFORCEMENT OPTIMIZATION PIPELINE ---
single_bar_results = []
for d in candidate_dias:
    res = solve_layout(d, bundled=False)
    if res is not None:
        single_bar_results.append(res)

if single_bar_results:
    active_layout = min(single_bar_results, key=lambda x: x["penalty"])
else:
    bundled_results = []
    for d in candidate_dias:
        b_res = solve_layout(d, bundled=True)
        if b_res is not None:
            bundled_results.append(b_res)
    if bundled_results:
        active_layout = min(bundled_results, key=lambda x: x["penalty"])
    else:
        active_layout = None

if active_layout is None:
    st.error("Column geometry cannot accommodate reinforcement demand within spacing limits (50 mm - 150 mm). Please increase B or D.")
    st.stop()

use_Bundle = active_layout["bundled"]
dia = active_layout["dia"]
de = active_layout["de"]
Nx = active_layout["Nx"]
Ny = active_layout["Ny"]
sx = active_layout["sx"]
sy = active_layout["sy"]

station_width = (2 * dia) if use_Bundle else dia
cc_x = sx + station_width
cc_y = sy + station_width

area_single_bar = int(np.round((np.pi * (dia ** 2)) / 4.0))
area_etabs_unit = int(np.round(2 * area_single_bar)) if use_Bundle else area_single_bar
etabs_bar_name = f"{dia}B" if use_Bundle else f"{dia}"

# --- SAFE DETERMINISTIC SUB-HOOP SCHEDULER ---
def get_sub_hoop_pairs(N):
    if N <= 2:
        return []
    if N == 3:
        return [(1, 1)]
    if N == 4:
        return [(1, 2)]
    if N == 5:
        return [(1, 3)]

    pairs = []
    for left in range(2, N // 2 + 1, 2):
        right = N - 1 - left
        if left <= right:
            pairs.append((left, right))

    if not pairs:
        pairs.append((1, N - 2))

    return sorted(list(set(pairs)))

vertical_sub_hoops = get_sub_hoop_pairs(Nx)
horizontal_sub_hoops = get_sub_hoop_pairs(Ny)

total_hoops_count = 1 + len(vertical_sub_hoops) + len(horizontal_sub_hoops)
tie_callout_image = f"{total_hoops_count} Φ {stirrup_dia}"

# --- UI DASHBOARD ---
col1, col2 = st.columns([1.15, 1.25])

with col1:
    st.subheader("Design Decision Summary")
    st.markdown(f"**Required Steel Area ($A_{{st}}$):** `{Ast_req:.1f} mm²` &nbsp;(**{p:.2f}%**)")
    st.markdown(
        f"**Optimized Provided Area:** `{active_layout['Ast_provided']:.1f} mm²` "
        f"&nbsp;(**{active_layout['p_provided']:.2f}%**)"
    )
    bundle_label = " (2-Bar Bundled)" if use_Bundle else " (Single Regular Bars)"
    st.markdown(
        f"**Reinforcement Provided:** **{active_layout['total_bars']}, {dia} mm Ø @ {int(np.round(cc_x))} mm c/c**{bundle_label}"
    )
    st.markdown(f"**Clear Spacing ($s_x, s_y$):** `{sx:.1f} mm, {sy:.1f} mm`")
    st.markdown(f"**Center-to-Center Spacing:** `{cc_x:.1f} mm, {cc_y:.1f} mm`")

    st.markdown("---")
    st.markdown("### 📋 ETABS Longitudinal Reinforcing Data")
    st.caption("Directly input these values into your ETABS Section Designer dialog:")

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border: 1px solid #d3d3d3; padding: 16px; border-radius: 6px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                <table style="width: 100%; font-size: 14px; border-collapse: separate; border-spacing: 0 10px;">
                    <tr>
                        <td style="color: #212529; width: 62%;"><strong>Number of Longitudinal Bars Along 3-dir Face</strong></td>
                        <td style="width: 38%;"><input type="text" value="{Nx}" disabled style="width: 100%; padding: 5px 8px; border: 1px solid #ced4da; border-radius: 4px; background-color: #ffffff; color: #212529; font-weight: bold; text-align: right;"></td>
                    </tr>
                    <tr>
                        <td style="color: #212529;"><strong>Number of Longitudinal Bars Along 2-dir Face</strong></td>
                        <td><input type="text" value="{Ny}" disabled style="width: 100%; padding: 5px 8px; border: 1px solid #ced4da; border-radius: 4px; background-color: #ffffff; color: #212529; font-weight: bold; text-align: right;"></td>
                    </tr>
                    <tr>
                        <td style="color: #212529;"><strong>Longitudinal Bar Size and Area</strong></td>
                        <td>
                            <div style="display: flex; gap: 6px;">
                                <input type="text" value="{etabs_bar_name}" disabled style="width: 45%; padding: 5px 8px; border: 1px solid #ced4da; border-radius: 4px; background-color: #ffffff; color: #212529; font-weight: bold; text-align: center;">
                                <input type="text" value="{area_etabs_unit} mm²" disabled style="width: 55%; padding: 5px 8px; border: 1px solid #ced4da; border-radius: 4px; background-color: #ffffff; color: #212529; font-weight: bold; text-align: right;">
                            </div>
                        </td>
                    </tr>
                    <tr>
                        <td style="color: #212529;"><strong>Corner Bar Size and Area</strong></td>
                        <td>
                            <div style="display: flex; gap: 6px;">
                                <input type="text" value="{etabs_bar_name}" disabled style="width: 45%; padding: 5px 8px; border: 1px solid #ced4da; border-radius: 4px; background-color: #ffffff; color: #212529; font-weight: bold; text-align: center;">
                                <input type="text" value="{area_etabs_unit} mm²" disabled style="width: 55%; padding: 5px 8px; border: 1px solid #ced4da; border-radius: 4px; background-color: #ffffff; color: #212529; font-weight: bold; text-align: right;">
                            </div>
                        </td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        if use_Bundle:
            st.info(
                f"ℹ️ **ETABS Bundling Parameter:** Select equivalent bar **`{etabs_bar_name}`** "
                f"(Area = **`{area_etabs_unit} mm²`**, $d_e$ = **`{de} mm`**)."
            )

# --- CROSS-SECTION CANVAS ---
with col2:
    st.subheader("Column Cross-Section View")
    fig, ax = plt.subplots(figsize=(9, 9))

    concrete = patches.Rectangle((0, 0), B, D, linewidth=2.2, edgecolor='black', facecolor='white', zorder=1)
    ax.add_patch(concrete)

    tie_ox = cover
    tie_oy = cover
    tie_ow = B - 2 * cover
    tie_oh = D - 2 * cover
    outer_tie = patches.Rectangle((tie_ox, tie_oy), tie_ow, tie_oh,
                                  linewidth=2.0, edgecolor='#000000', facecolor='none', zorder=2)
    ax.add_patch(outer_tie)

    hook_len = max(6 * stirrup_dia, 60)
    ax.plot([tie_ox + hook_len, tie_ox], [tie_oy + tie_oh - hook_len, tie_oy + tie_oh], color='#000000', linewidth=2.0, zorder=2)
    ax.plot([tie_ox, tie_ox + hook_len], [tie_oy + tie_oh, tie_oy + tie_oh - hook_len], color='#000000', linewidth=2.0, zorder=2)

    tie_ix = cover + stirrup_dia
    tie_iy = cover + stirrup_dia
    tie_iw = B - 2 * (cover + stirrup_dia)
    tie_ih = D - 2 * (cover + stirrup_dia)

    r = dia / 2.0
    x_min = tie_ix + r
    x_max = tie_ix + tie_iw - r
    y_min = tie_iy + r
    y_max = tie_iy + tie_ih - r

    xs = np.linspace(x_min, x_max, Nx)
    ys = np.linspace(y_min, y_max, Ny)

    target_link_center = None

    for (i1, i2) in vertical_sub_hoops:
        if i1 == i2:
            cx = xs[i1]
            ax.plot([cx, cx], [tie_oy, tie_oy + tie_oh], color='#00529B', linewidth=1.6, zorder=3)
            ax.plot([cx - 25, cx], [tie_oy + tie_oh - 25, tie_oy + tie_oh], color='#00529B', linewidth=1.6, zorder=3)
            ax.plot([cx, cx + 25], [tie_oy, tie_oy + 25], color='#00529B', linewidth=1.6, zorder=3)
            if target_link_center is None:
                target_link_center = (cx, tie_oy + tie_oh * 0.35)
        else:
            vx = xs[i1] - r - stirrup_dia
            vw = (xs[i2] + r + stirrup_dia) - vx
            vy = tie_oy
            vh = tie_oh
            rect_v = patches.Rectangle((vx, vy), vw, vh, linewidth=1.6, edgecolor='#00529B', facecolor='none', zorder=3)
            ax.add_patch(rect_v)
            ax.plot([vx + 35, vx], [vy + vh - 35, vy + vh], color='#00529B', linewidth=1.6, zorder=3)
            ax.plot([vx, vx + 35], [vy + vh, vy + vh - 35], color='#00529B', linewidth=1.6, zorder=3)
            if target_link_center is None:
                target_link_center = (vx + vw / 2.0, vy + vh * 0.35)

    for (j1, j2) in horizontal_sub_hoops:
        if j1 == j2:
            cy = ys[j1]
            ax.plot([tie_ox, tie_ox + tie_ow], [cy, cy], color='#008000', linewidth=1.6, zorder=3)
            ax.plot([tie_ox + 25, tie_ox], [cy + 25, cy], color='#008000', linewidth=1.6, zorder=3)
            ax.plot([tie_ox + tie_ow - 25, tie_ox + tie_ow], [cy, cy - 25], color='#008000', linewidth=1.6, zorder=3)
        else:
            hx = tie_ox
            hw = tie_ow
            hy = ys[j1] - r - stirrup_dia
            hh = (ys[j2] + r + stirrup_dia) - hy
            rect_h = patches.Rectangle((hx, hy), hw, hh, linewidth=1.6, edgecolor='#008000', facecolor='none', zorder=3)
            ax.add_patch(rect_h)
            ax.plot([hx + 35, hx], [hy + hh - 35, hy + hh], color='#008000', linewidth=1.6, zorder=3)
            ax.plot([hx, hx + 35], [hy + hh, hy + hh - 35], color='#008000', linewidth=1.6, zorder=3)

    stations = []
    for x in xs[1:-1]:
        stations.append((x, y_min, 'bottom'))
        stations.append((x, y_max, 'top'))
    for y in ys[1:-1]:
        stations.append((x_min, y, 'left'))
        stations.append((x_max, y, 'right'))

    corners = [
        (x_min, y_min, 'corner_bl'),
        (x_max, y_min, 'corner_br'),
        (x_min, y_max, 'corner_tl'),
        (x_max, y_max, 'corner_tr')
    ]
    stations.extend(corners)

    diag_shift = dia / np.sqrt(2)
    sample_corner_bar = (x_max, y_max)

    for x, y, pos in stations:
        if not use_Bundle:
            circle = patches.Circle((x, y), r, facecolor='black', edgecolor='black', linewidth=1, zorder=5)
            ax.add_patch(circle)
        else:
            if pos == 'corner_bl':
                c1, c2 = (x, y), (x + diag_shift, y + diag_shift)
            elif pos == 'corner_br':
                c1, c2 = (x, y), (x - diag_shift, y + diag_shift)
            elif pos == 'corner_tl':
                c1, c2 = (x, y), (x + diag_shift, y - diag_shift)
            elif pos == 'corner_tr':
                c1, c2 = (x, y), (x - diag_shift, y - diag_shift)
            elif pos in ['bottom', 'top']:
                c1, c2 = (x - r, y), (x + r, y)
            elif pos in ['left', 'right']:
                c1, c2 = (x, y - r), (x, y + r)

            ax.add_patch(patches.Circle(c1, r, facecolor='black', edgecolor='black', linewidth=1, zorder=5))
            ax.add_patch(patches.Circle(c2, r, facecolor='black', edgecolor='black', linewidth=1, zorder=5))

    bundle_str_top = " (BUNDLED)" if use_Bundle else ""
    rebar_callout_top = f"{active_layout['total_bars']} Φ {dia}{bundle_str_top}"
    ax.annotate(
        rebar_callout_top,
        xy=sample_corner_bar,
        xytext=(B * 1.05, D * 1.12),
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
        fontsize=12,
        fontweight='bold',
        color='black'
    )
    ax.plot([B * 1.03, B * 1.38], [D * 1.09, D * 1.09], color='black', lw=1.2)

    if target_link_center is None:
        target_link_center = (xs[Nx // 2], ys[Ny // 2])
    ax.annotate(
        tie_callout_image,
        xy=target_link_center,
        xytext=(B * 1.05, D * 0.25),
        arrowprops=dict(
            arrowstyle='->',
            color='black',
            lw=1.2,
            connectionstyle="angle,angleA=0,angleB=90,rad=0"
        ),
        fontsize=12,
        fontweight='bold',
        color='black'
    )
    ax.plot([B * 1.03, B * 1.35], [D * 0.22, D * 0.22], color='black', lw=1.2)

    dim_off_y = D + D * 0.08
    dim_off_x = -B * 0.12
    ax.annotate('', xy=(0, dim_off_y), xytext=(B, dim_off_y), arrowprops=dict(arrowstyle='-', color='black', lw=1.1))
    ax.plot([-15, 15], [dim_off_y - 15, dim_off_y + 15], color='black', lw=1.3)
    ax.plot([B - 15, B + 15], [dim_off_y - 15, dim_off_y + 15], color='black', lw=1.3)
    ax.text(B / 2.0, dim_off_y + D * 0.03, f"B = {B} mm", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.annotate('', xy=(dim_off_x, 0), xytext=(dim_off_x, D), arrowprops=dict(arrowstyle='-', color='black', lw=1.1))
    ax.plot([dim_off_x - 15, dim_off_x + 15], [-15, 15], color='black', lw=1.3)
    ax.plot([dim_off_x - 15, dim_off_x + 15], [D - 15, D + 15], color='black', lw=1.3)
    ax.text(dim_off_x - B * 0.04, D / 2.0, f"D = {D} mm", ha='right', va='center', rotation=90, fontsize=11, fontweight='bold')

    ax.set_xlim(-B * 0.25, B * 1.45)
    ax.set_ylim(-D * 0.10, D * 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    st.pyplot(fig)
    plt.close(fig)
