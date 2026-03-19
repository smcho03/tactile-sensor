"""
sanity_utils.py
───────────────
공통 시각화 헬퍼

축 convention:
  h 패널    : 가로=y [um],  세로=x [um]
  CMOS 패널 : 가로=y'[um],  세로=z'[um]

위상 표시 정책:
  - 개별 위상  : unwrap_phase(np.angle(U))
  - delta_phi  : np.angle(U_def * conj(U_ref)) → unwrap_phase
                 (wrapped 뺄셈 금지: phi_def - phi_ref 는 [-2π,2π]로 unwrap 오판 유발)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from skimage.restoration import unwrap_phase

STYLE = {
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#0d1117',
    'text.color': '#e6edf3', 'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'font.family': 'monospace',
}

def _cb(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, color='#8b949e', fontsize=7)
    cb.ax.yaxis.set_tick_params(color='#8b949e', labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e')

def _grid(ax):
    ax.grid(True, color='#30363d', linewidth=0.4, linestyle=':')
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')


def plot_h(fig, ax, h_nm, x_coords, y_coords, title):
    x_um, y_um = x_coords*1e6, y_coords*1e6
    vmin = min(h_nm.min(), -1e-6)
    vmax = max(h_nm.max(),  1e-6)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(h_nm.T,
                   extent=[y_um.min(), y_um.max(), x_um.min(), x_um.max()],
                   origin='lower', aspect='auto', cmap='RdBu', norm=norm)
    _cb(fig, ax, im, 'h [nm]')
    ax.set_title(title); ax.set_xlabel('y [um]'); ax.set_ylabel('x [um]')
    _grid(ax)


def plot_intensity(fig, ax, I, y_prime, z_prime, title, vmax=None):
    yp_um, zp_um = y_prime*1e6, z_prime*1e6
    im = ax.imshow(I,
                   extent=[yp_um.min(), yp_um.max(), zp_um.min(), zp_um.max()],
                   origin='lower', aspect='auto', cmap='inferno',
                   vmin=0, vmax=vmax if vmax else I.max())
    _cb(fig, ax, im, 'I [a.u.]')
    ax.set_title(title); ax.set_xlabel("y' [um]"); ax.set_ylabel("z' [um]")
    _grid(ax)


def plot_phase(fig, ax, phi_raw, y_prime, z_prime, title, sym=False):
    """
    phi_raw : np.angle(U) 또는 np.angle(U_def * conj(U_ref))
              → 반드시 [-π, π] 범위여야 unwrap이 정상 동작
    sym     : True → RdBu 대칭 (delta_phi 패널 전용)
    """
    yp_um, zp_um = y_prime*1e6, z_prime*1e6

    phi = unwrap_phase(phi_raw)
    span = phi.max() - phi.min()
    vmin_d, vmax_d = float(phi.min()), float(phi.max())

    if sym:
        abs_max = max(abs(vmin_d), abs(vmax_d), 1e-12)
        cmap, vmin, vmax = 'RdBu', -abs_max, abs_max
        cb_label = 'Δφ [rad]  (unwrapped)'
    elif span <= 2*np.pi + 1e-6:
        cmap, vmin, vmax = 'hsv', -np.pi, np.pi
        cb_label = 'φ [rad]  (≤2π, hsv)'
    else:
        cmap, vmin, vmax = 'viridis', vmin_d, vmax_d
        cb_label = f'φ [rad]  (unwrapped, span={span:.1f})'

    im = ax.imshow(phi,
                   extent=[yp_um.min(), yp_um.max(), zp_um.min(), zp_um.max()],
                   origin='lower', aspect='auto',
                   cmap=cmap, vmin=vmin, vmax=vmax)

    if cmap == 'hsv':
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels(['-π', '-π/2', '0', '+π/2', '+π'],
                              color='#8b949e', fontsize=6)
        cb.set_label(cb_label, color='#8b949e', fontsize=7)
    else:
        _cb(fig, ax, im, cb_label)

    ax.set_title(title); ax.set_xlabel("y' [um]"); ax.set_ylabel("z' [um]")
    _grid(ax)


# ─────────────────────────────────────────────────────────────
def plot_1x3(title, h_nm, x_coords, y_coords,
             I, phi, y_prime, z_prime, out_path):
    """1행 x 3열: h | I_CMOS | Phase_CMOS (unwrapped)
    phi = np.angle(U_CMOS) 를 그대로 넘길 것
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=11, color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    plot_h(fig, axes[0], h_nm, x_coords, y_coords, 'Surface h(x,y)')
    plot_intensity(fig, axes[1], I, y_prime, z_prime, 'CMOS Intensity')
    plot_phase(fig, axes[2], phi, y_prime, z_prime, 'CMOS Phase  (unwrapped)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> {out_path.name}")


def plot_3x3(title,
             h_ref_nm, h_def_nm, x_coords, y_coords,
             I_ref, I_def,
             U_ref_cmos, U_def_cmos,       # ← phi 대신 복소장 U 로 변경
             y_prime, z_prime,
             out_path):
    """
    3행 x 3열:
      row0: h=0     | I_ref  | phi_ref  (unwrapped)
      row1: h_def   | I_def  | phi_def  (unwrapped)
      row2: delta_h | dI     | delta_phi (올바른 wrapped 차이 → unwrap)

    delta_phi 계산:
      np.angle(U_def * conj(U_ref))  →  정확히 [-π,π] wrap된 위상차
      → unwrap_phase()               →  연속 위상차
      (phi_def - phi_ref 단순 뺄셈은 [-2π,2π]가 되어 unwrap 오판 발생)
    """
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=11, color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    phi_ref = np.angle(U_ref_cmos)
    phi_def = np.angle(U_def_cmos)

    # ★ 핵심: 복소곱으로 위상차를 [-π,π]로 정확히 wrap
    dphi_wrapped = np.angle(U_def_cmos * np.conj(U_ref_cmos))

    I_max    = max(I_ref.max(), I_def.max())
    dI       = I_def - I_ref
    dh_nm    = h_def_nm - h_ref_nm
    yp_um    = y_prime * 1e6
    zp_um    = z_prime * 1e6
    dI_min   = min(dI.min(), -1e-20)
    dI_max_v = max(dI.max(),  1e-20)

    # ── row 0: reference ─────────────────────────────────────
    plot_h(fig, axes[0,0], h_ref_nm, x_coords, y_coords, 'h = 0  (reference)')
    plot_intensity(fig, axes[0,1], I_ref, y_prime, z_prime,
                   'CMOS Intensity  (ref)', vmax=I_max)
    plot_phase(fig, axes[0,2], phi_ref, y_prime, z_prime,
               'CMOS Phase  (ref, unwrapped)')

    # ── row 1: deformed ──────────────────────────────────────
    plot_h(fig, axes[1,0], h_def_nm, x_coords, y_coords, 'h  (deformed)')
    plot_intensity(fig, axes[1,1], I_def, y_prime, z_prime,
                   'CMOS Intensity  (def)', vmax=I_max)
    plot_phase(fig, axes[1,2], phi_def, y_prime, z_prime,
               'CMOS Phase  (def, unwrapped)')

    # ── row 2: difference ────────────────────────────────────
    plot_h(fig, axes[2,0], dh_nm, x_coords, y_coords, 'delta_h  (def - ref)')

    ax = axes[2,1]
    im = ax.imshow(dI,
                   extent=[yp_um.min(), yp_um.max(), zp_um.min(), zp_um.max()],
                   origin='lower', aspect='auto', cmap='RdBu',
                   norm=TwoSlopeNorm(vmin=dI_min, vcenter=0, vmax=dI_max_v))
    _cb(fig, ax, im, 'dI [a.u.]')
    ax.set_title('delta_I  (def - ref)')
    ax.set_xlabel("y' [um]"); ax.set_ylabel("z' [um]")
    _grid(ax)

    # ★ dphi_wrapped ([-π,π]) → unwrap → 연속 위상차
    plot_phase(fig, axes[2,2], dphi_wrapped, y_prime, z_prime,
               'delta_phi  (unwrapped)', sym=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> {out_path.name}")