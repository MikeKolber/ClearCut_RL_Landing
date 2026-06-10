"""Interactive RL-vs-ASTOS comparison with cursor readout.

Usage: python view_comparison.py [ppo|sac]

Opens a matplotlib window with a crosshair + value tooltip that follows
the mouse, showing interpolated ASTOS and RL values at the cursor position.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from landing_env import RocketLandingEnv
from rollout import run_episode, ALGOS

ASTOS_FILE = "Full_data_4Michael.xlsx"
MODELS_DIR = os.path.join("output", "models")


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "ppo"

    df = pd.read_excel(ASTOS_FILE, header=None)
    df.columns = ["time", "altitude", "velocity", "thrust", "mass"]
    df["altitude"] *= 1000
    df["velocity"] *= 1000
    df["thrust"]   *= 1000
    df["mass"]     *= 1000

    apogee_idx = int(df["altitude"].idxmax())
    t_apogee = df.loc[apogee_idx, "time"]
    a_t, a_h, a_v, a_thr, a_m = [df[c].values for c in df.columns]

    model = ALGOS[prefix].load(os.path.join(MODELS_DIR, f"{prefix}_lander"))
    env = RocketLandingEnv()
    rl_data, rl_summary = run_episode(model, env)

    rl_t, rl_h, rl_v, rl_thr, rl_m = rl_data.T
    rl_t_shifted = rl_t + t_apogee
    rl_thrust_N = rl_thr * config.MAX_THRUST

    label = prefix.upper()
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 13))
    astos_kw = dict(color="C3", linewidth=1.5, linestyle=":", label="ASTOS (full flight)")
    rl_kw    = dict(color="C0", linewidth=2, label=f"{label} RL (landing)")

    axes[0].plot(a_t, a_h, **astos_kw)
    axes[0].plot(rl_t_shifted, rl_h, **rl_kw)
    axes[0].axvline(t_apogee, ls=":", color="grey", alpha=0.6, label="apogee")
    axes[0].set_ylabel("altitude [m]")
    axes[0].legend(loc="upper right")

    axes[1].plot(a_t, a_v, **astos_kw)
    axes[1].plot(rl_t_shifted, rl_v, **rl_kw)
    axes[1].axvline(t_apogee, ls=":", color="grey", alpha=0.6)
    axes[1].set_ylabel("velocity [m/s]")

    axes[2].plot(a_t, a_thr, **astos_kw)
    axes[2].plot(rl_t_shifted, rl_thrust_N, **rl_kw)
    axes[2].axvline(t_apogee, ls=":", color="grey", alpha=0.6)
    axes[2].set_ylabel("thrust [N]")

    axes[3].plot(a_t, a_m, **astos_kw)
    axes[3].plot(rl_t_shifted, rl_m, **rl_kw)
    axes[3].axvline(t_apogee, ls=":", color="grey", alpha=0.6)
    axes[3].set_ylabel("mass [kg]")
    axes[3].set_xlabel("time [s]")

    apogee_row = df.loc[apogee_idx]
    ic_text = (f"ASTOS at apogee (v=0):\n"
               f"  altitude = {apogee_row['altitude']:.1f} m\n"
               f"  velocity = {apogee_row['velocity']:.4f} m/s\n"
               f"  thrust   = {apogee_row['thrust']:.0f} N\n"
               f"  fuel     = {apogee_row['mass'] - config.DRY_MASS:.1f} kg")
    axes[3].text(0.02, 0.05, ic_text, transform=axes[3].transAxes,
                 fontsize=9, verticalalignment="bottom", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    for ax in axes:
        ax.grid(True, alpha=0.3)

    impact_str = f"{rl_summary['impact']:.2f}" if not np.isnan(rl_summary['impact']) else "n/a"
    axes[0].set_title(f"ASTOS vs {label} RL  "
                      f"({rl_summary['outcome']}, impact {impact_str} m/s)")

    fig.tight_layout()

    # --- interactive cursor ---
    vlines = [ax.axvline(0, ls='-', lw=0.8, color='black', alpha=0.4, visible=False)
              for ax in axes]
    annot = fig.text(0, 0, '', fontsize=8, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                               alpha=0.95, ec='grey'),
                     visible=False, zorder=100)

    def on_move(event):
        if event.inaxes not in axes:
            for vl in vlines:
                vl.set_visible(False)
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

        t = event.xdata
        for vl in vlines:
            vl.set_xdata([t])
            vl.set_visible(True)

        lines = [f"t = {t:.2f} s", ""]
        if a_t[0] <= t <= a_t[-1]:
            lines.append(f"ASTOS  h={np.interp(t,a_t,a_h):7.1f}m  "
                         f"v={np.interp(t,a_t,a_v):7.2f}m/s  "
                         f"T={np.interp(t,a_t,a_thr):7.0f}N  "
                         f"m={np.interp(t,a_t,a_m):7.1f}kg")
        else:
            lines.append("ASTOS  (out of range)")

        if len(rl_t_shifted) > 0 and rl_t_shifted[0] <= t <= rl_t_shifted[-1]:
            lines.append(f"RL     h={np.interp(t,rl_t_shifted,rl_h):7.1f}m  "
                         f"v={np.interp(t,rl_t_shifted,rl_v):7.2f}m/s  "
                         f"T={np.interp(t,rl_t_shifted,rl_thrust_N):7.0f}N  "
                         f"m={np.interp(t,rl_t_shifted,rl_m):7.1f}kg")
        else:
            lines.append("RL     (out of range)")

        annot.set_text('\n'.join(lines))
        fig_x, fig_y = fig.transFigure.inverted().transform((event.x, event.y))
        annot.set_position((min(fig_x + 0.01, 0.65), min(fig_y + 0.01, 0.95)))
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()


if __name__ == "__main__":
    main()
