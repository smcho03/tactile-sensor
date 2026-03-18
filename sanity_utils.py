"""
sanity_utils.py
───────────────
공통 시각화 헬퍼

축 convention:
  h 패널    : 가로=y [um],  세로=x [um]
  CMOS 패널 : 가로=y'[um],  세로=z'[um]
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

# ── 공통 스타일 ───────────────────────────────────────────────
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
    """
    h(x,y) 패널
    가로=y [um],  세로=x [um]
    h.shape=(Ny,Nx) → .T → shape=(Nx,Ny): imshow 가로=y, 세로=x
    """
    x_um, y_um = x_coords*1e6, y_coords*1e6
    h_abs = max(np.abs(h_nm).max(), 1e-6)
    vmin  = min(h_nm.min(), -1e-6)
    vmax  = max(h_nm.max(),  1e-6)
    norm  = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(h_nm.T,
                   extent=[y_um.min(), y_um.max(), x_um.min(), x_um.max()],
                   origin='lower', aspect='auto', cmap='RdBu', norm=norm)
    _cb(fig, ax, im, 'h [nm]')
    ax.set_title(title)
    ax.set_xlabel('y [um]')
    ax.set_ylabel('x [um]')
    _grid(ax)


def plot_intensity(fig, ax, I, y_prime, z_prime, title, vmax=None):
    """
    CMOS 세기 패널
    가로=y'[um],  세로=z'[um]
    I.shape=(Nz,Ny) → imshow origin=lower: 가로=y', 세로=z'
    """
    yp_um, zp_um = y_prime*1e6, z_prime*1e6
    im = ax.imshow(I,
                   extent=[yp_um.min(), yp_um.max(), zp_um.min(), zp_um.max()],
                   origin='lower', aspect='auto', cmap='inferno',
                   vmin=0, vmax=vmax if vmax else I.max())
    _cb(fig, ax, im, 'I [a.u.]')
    ax.set_title(title)
    ax.set_xlabel("y' [um]")
    ax.set_ylabel("z' [um]")
    _grid(ax)


def plot_phase(fig, ax, phi, y_prime, z_prime, title,
               cmap='hsv', vmin=-np.pi, vmax=np.pi):
    """
    CMOS 위상 패널
    가로=y'[um],  세로=z'[um]
    phi.shape=(Nz,Ny) → imshow origin=lower: 가로=y', 세로=z'
    """
    yp_um, zp_um = y_prime*1e6, z_prime*1e6
    im = ax.imshow(phi,
                   extent=[yp_um.min(), yp_um.max(), zp_um.min(), zp_um.max()],
                   origin='lower', aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax)
    if cmap == 'hsv':
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels(['-pi', '-pi/2', '0', '+pi/2', '+pi'],
                              color='#8b949e', fontsize=6)
        cb.set_label('phi [rad]', color='#8b949e', fontsize=7)
    else:
        _cb(fig, ax, im, 'dphi [rad]')
    ax.set_title(title)
    ax.set_xlabel("y' [um]")
    ax.set_ylabel("z' [um]")
    _grid(ax)


def plot_1x3(title, h_nm, x_coords, y_coords,
             I, phi, y_prime, z_prime, out_path):
    """1행 x 3열: h | I_CMOS | Phase_CMOS"""
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=11, color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    plot_h(fig, axes[0], h_nm, x_coords, y_coords, 'Surface h(x,y)')
    plot_intensity(fig, axes[1], I, y_prime, z_prime, 'CMOS Intensity')
    plot_phase(fig, axes[2], phi, y_prime, z_prime, 'CMOS Phase')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> {out_path.name}")


def plot_3x3(title,
             h_ref_nm, h_def_nm, x_coords, y_coords,
             I_ref, I_def,
             phi_ref, phi_def,
             y_prime, z_prime,
             out_path):
    """
    3행 x 3열:
      row0: h=0      | I_ref    | phi_ref
      row1: h_def    | I_def    | phi_def
      row2: delta_h  | delta_I  | delta_phi (unwrapped)
    """
    from skimage.restoration import unwrap_phase

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(title, fontsize=11, color='#e6edf3')
    fig.patch.set_facecolor('#0d1117')

    I_max    = max(I_ref.max(), I_def.max())
    dI       = I_def - I_ref
    dphi     = unwrap_phase(phi_def - phi_ref)
    dh_nm    = h_def_nm - h_ref_nm
    yp_um    = y_prime * 1e6
    zp_um    = z_prime * 1e6

    dphi_abs = max(np.abs(dphi).max(), 1e-12)
    dI_min   = min(dI.min(), -1e-20)
    dI_max_v = max(dI.max(),  1e-20)

    # row 0: reference
    plot_h(fig, axes[0,0], h_ref_nm, x_coords, y_coords, 'h = 0  (reference)')
    plot_intensity(fig, axes[0,1], I_ref, y_prime, z_prime,
                   'CMOS Intensity  (ref)', vmax=I_max)
    plot_phase(fig, axes[0,2], phi_ref, y_prime, z_prime,
               'CMOS Phase  (ref)')

    # row 1: deformed
    plot_h(fig, axes[1,0], h_def_nm, x_coords, y_coords, 'h  (deformed)')
    plot_intensity(fig, axes[1,1], I_def, y_prime, z_prime,
                   'CMOS Intensity  (def)', vmax=I_max)
    plot_phase(fig, axes[1,2], phi_def, y_prime, z_prime,
               'CMOS Phase  (def)')

    # row 2: difference
    plot_h(fig, axes[2,0], dh_nm, x_coords, y_coords, 'delta_h  (def - ref)')

    # delta_I
    ax = axes[2,1]
    im = ax.imshow(dI,
                   extent=[yp_um.min(), yp_um.max(), zp_um.min(), zp_um.max()],
                   origin='lower', aspect='auto', cmap='RdBu',
                   norm=TwoSlopeNorm(vmin=dI_min, vcenter=0, vmax=dI_max_v))
    _cb(fig, ax, im, 'dI [a.u.]')
    ax.set_title('delta_I  (def - ref)')
    ax.set_xlabel("y' [um]"); ax.set_ylabel("z' [um]")
    _grid(ax)

    # delta_phi unwrapped
    plot_phase(fig, axes[2,2], dphi, y_prime, z_prime,
               'delta_phi  (unwrapped)',
               cmap='RdBu', vmin=-dphi_abs, vmax=dphi_abs)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  saved -> {out_path.name}")