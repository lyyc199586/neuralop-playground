"""
plot.py

This module provides various visualization utilities for plotting and animating
2D and 3D FNO input and output data, including ground truth and predictions.
It supports both static and animated visualizations, as well as voxel-based 3D plotting.

Author: Yangyuanchen Liu
Date: 2025-04-04

Overview:
- Loads RAMPs and spatiotemporal variable `.npy` files from FEM results directories.
- Stacks and reshapes data into `torch.Tensor` formats.
- Saves preprocessed data as `.pt` files with structured naming.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter
from typing import Optional, Tuple, List, Union
from sklearn.metrics import mean_squared_error


def data_image(
    x: np.ndarray,
    y: np.ndarray,
    t: int,
    axes: Optional[np.ndarray] = None,
    show_colorbar: bool = True,
) -> np.ndarray:
    """
    Visualize 2x3 subplots of input (x) and output (y) data at a specific time step.

    Parameters:
    - x: np.ndarray, shape (3, height, width, time)
        Input data with 3 channels.
    - y: np.ndarray, shape (3, height, width, time)
        Output data with 3 channels.
    - t: int
        Time step to visualize.
    - axes: Optional[np.ndarray]
        Predefined axes for plotting. If None, new axes are created.
    - show_colorbar: bool
        Whether to display colorbars for each subplot.

    Returns:
    - axes: np.ndarray
        Axes used for the plots.
    """

    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    else:
        fig = axes[0, 0].figure

    # plot x: Ek(x,t); M(x); RAMPs(x)
    axes[0, 0].annotate(
        f"x",
        xy=(-0.15, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
    )

    Ek = np.ma.masked_where(x[0, :, :, t] == 0, x[0, :, :, t])
    viridis_white = plt.cm.viridis.copy()
    viridis_white.set_bad(color="white")

    im0 = axes[0, 0].imshow(Ek, cmap=viridis_white, origin="lower")
    axes[0, 0].set_title("Sensor signal $E_k(x, t)$")

    im1 = axes[0, 1].imshow(x[1, :, :, t], cmap="gray_r", origin="lower")
    axes[0, 1].set_title("Mask $M(x)$")

    im2 = axes[0, 2].imshow(x[2, :, :, t], origin="lower")
    axes[0, 2].set_title("RAMPs field $\Phi(x)$")

    # plot y: E(x, t); F(x, t); D(x, t)
    axes[1, 0].annotate(
        f"y",
        xy=(-0.15, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
    )
    im3 = axes[1, 0].imshow(y[0, :, :, t], origin="lower")
    axes[1, 0].set_title("Electricity $E(x, t)$")

    im4 = axes[1, 1].imshow(y[1, :, :, t], origin="lower")
    axes[1, 1].set_title("Defgrad $F(x, t)$")

    im5 = axes[1, 2].imshow(y[2, :, :, t], origin="lower")
    axes[1, 2].set_title("Damage $D(x, t)$")

    # clean ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # set colorbar
    if show_colorbar:
        ims = [im0, im1, im2, im3, im4, im5]
        for ax, im in zip(axes.flat, ims):
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.07, location="bottom")

    plt.tight_layout()
    plt.show()

    return axes

def sci_label(value: float, prefix: str = None) -> str:
    """
    Format a number in scientific notation with a given prefix.
    """
    base, exp = f"{value:.2e}".split("e")
    if prefix:
        return rf"$\mathrm{{{prefix}}}: {base} \times 10^{{{int(exp)}}}$"
    else:
        return rf"${base} \times 10^{{{int(exp)}}}$"


def pred_plot(
    y: np.ndarray,
    pred: np.ndarray,
    t: int,
    show_colorbar: bool = True,
    vmins: Optional[List[float]] = None,
    vmaxs: Optional[List[float]] = None,
    plot_method: str = "image",
    level: int = 21,
    cmap: str = "viridis",
    show_error: bool = False,
    error_cmap: str = "coolwarm",
) -> Tuple[np.ndarray, List]:
    """
    Plot ground truth and predictions in a 2x3 grid.
    if show_error is True, plot error in the last row.

    Parameters:
    - y: np.ndarray, shape (3, height, width, time)
        Ground truth data.
    - pred: np.ndarray, shape (3, height, width, time)
        Predicted data.
    - t: int
        Time step to visualize.
    - show_colorbar: bool
        Whether to display colorbars for each subplot.
    - vmins: Optional[List[float]]
        Minimum values for color normalization. If None, computed from data.
    - vmaxs: Optional[List[float]]
        Maximum values for color normalization. If None, computed from data.
    - plot_method: str
        Plotting method, either 'image' or 'contourf'.
    - level: int
        Number of contour levels for 'contourf' method.
    - cmap: str
        Colormap to use for plotting.
    - show_error: bool
        Whether to show error plots.
    - error_cmap: str
        Colormap for error plots.

    Returns:
    - axes: np.ndarray
        Axes used for the plots.
    - ims: List
        List of image objects for the plots.
    """
    # fig layout settings
    base_rows = 2
    extra_rows = 1 if show_error else 0
    cb_rows = 1 + (1 if show_error else 0) if show_colorbar else 0
    n_rows = base_rows + extra_rows + cb_rows
    height_ratios = [1, 1]
    
    if show_colorbar:
        height_ratios += [0.05]
    
    if show_error:
        height_ratios += [1]
        if show_colorbar:
            height_ratios += [0.05]

    total_heights = sum(height_ratios)
    fig_h = total_heights * 2.5
    fig_w = 7
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, 3, height_ratios=height_ratios, figure=fig)
    
    # row index settings
    ROW_GT = 0
    ROW_PRED = 1
    ROW_SHARED_CB = 2 if show_colorbar else None
    ROW_ERROR = 3 if show_error and show_colorbar else 2 if show_error else None
    ROW_ERROR_CB = 4 if show_error and show_colorbar else None
    axes = np.empty((n_rows, 3), dtype=object)
    
    # data settings
    if vmins is None:
        vmins = [min(y[i, :, :, t].min(), pred[i, :, :, t].min()) for i in range(3)]
    if vmaxs is None:
        vmaxs = [max(y[i, :, :, t].max(), pred[i, :, :, t].max()) for i in range(3)]

    titles = ["Electric Field", "Deformation Field", "Damage Indicator"]
    labels = [r"$\xi_3$ (V/m)", r"$F_{22}$", r"$I_D$"]
    norms = [mcolors.Normalize(vmin=vmins[i], vmax=vmaxs[i]) for i in range(3)]

    ims = []

    # row 0: Ground truth
    for i in range(3):
        ax = fig.add_subplot(gs[ROW_GT, i])
        axes[0, i] = ax
        data = y[i, :, :, t]
        if plot_method == "contourf":
            levels = np.linspace(vmins[i], vmaxs[i], level)
            im = ax.contourf(data, levels=levels, extend="both", cmap=cmap)
        else:
            im = ax.imshow(data, origin="lower", norm=norms[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    # row 1: Prediction
    for i in range(3):
        ax = fig.add_subplot(gs[ROW_PRED, i])
        axes[1, i] = ax
        data = pred[i, :, :, t]
        if plot_method == "contourf":
            levels = np.linspace(vmins[i], vmaxs[i], level)
            im = ax.contourf(data, levels=levels, extend="both", cmap=cmap)
        else:
            im = ax.imshow(data, origin="lower", norm=norms[i], cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
        
    # row 2: shared colorbars
    if show_colorbar:
        for i in range(3):
            cax = fig.add_subplot(gs[ROW_SHARED_CB, i])
            cbar = fig.colorbar(ims[i + 3], cax=cax, orientation="horizontal")
            cbar.set_label(labels[i])
            cbar.set_ticks([vmins[i], vmaxs[i]])

    # row 3: Error = Prediction - Ground truth
    if show_error:
        eps = 1e-12
        errors = pred - y
        rel_errors = errors / (y + eps)
        mses = [
            mean_squared_error(y[i, :, :, t].ravel(), pred[i, :, :, t].ravel())
            for i in range(3)
        ]
        error_norms = [
            mcolors.TwoSlopeNorm(
                vcenter=0.0,
                vmin=rel_errors[i, :, :, t].min(),
                vmax=rel_errors[i, :, :, t].max(),
            )
            for i in range(3)
        ]
        for i in range(3):
            ax = fig.add_subplot(gs[ROW_ERROR, i])
            axes[2, i] = ax
            data = rel_errors[i, :, :, t]
            if plot_method == "contourf":
                levels = np.linspace(
                    rel_errors[i, :, :, t].min(), rel_errors[i, :, :, t].max(), level
                )
                im = ax.contourf(data, levels=levels, extend="both", cmap=error_cmap)
            else:
                im = ax.imshow(data, origin="lower", norm=error_norms[i], cmap=error_cmap)
            ax.set_title(sci_label(value=mses[i], prefix="MSE"))
            ax.set_xticks([])
            ax.set_yticks([])
            ims.append(im)

        # row 4: error colorbars
        if show_colorbar:
            for i in range(3):
                cax = fig.add_subplot(gs[ROW_ERROR_CB, i])
                sm = plt.cm.ScalarMappable(cmap=error_cmap, norm=error_norms[i])
                sm.set_array([])
                
                vmin = rel_errors[i, :, :, t].min()
                vmax = rel_errors[i, :, :, t].max()
                cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([rf"${vmin*100:.1f}\%$", rf"${vmax*100:.1f}\%$"])
    # row annotation
    axes[0, 0].annotate("Ground truth", xy=(-0.05, 0.5), xycoords="axes fraction",
                    va="center", ha="right", rotation=90)

    axes[1, 0].annotate("Prediction", xy=(-0.05, 0.5), xycoords="axes fraction",
                    va="center", ha="right", rotation=90)

    if show_error:
        axes[2, 0].annotate("Relative Error", xy=(-0.05, 0.5), xycoords="axes fraction",
                            va="center", ha="right", rotation=90)

    return fig, axes, ims


def pred_anim(
    y: np.ndarray,
    pred: np.ndarray,
    t_range: Optional[range] = None,
    save_path: Optional[str] = None,
    fps: int = 2,
    **kwargs,
) -> None:
    """
    Create an animation of ground truth and predictions over a range of time steps.

    Parameters:
    - y: np.ndarray, shape (3, height, width, time)
        Ground truth data.
    - pred: np.ndarray, shape (3, height, width, time)
        Predicted data.
    - t_range: Optional[range]
        Range of time steps to animate.
    - save_path: Optional[str]
        Path to save the animation. If None, the animation is displayed.
    - fps: int
        Frames per second for the animation.
    - kwargs: dict
        Additional keyword arguments for `pred_plot` function.
    """
    if t_range is None:
        t_range = range(y.shape[-1])

    fig, _, ims = pred_plot(y, pred, t_range[0], **kwargs)

    def update(t):
        for i in range(3):
            ims[i].set_array(y[i, :, :, t])
            ims[i + 3].set_array(pred[i, :, :, t])
        fig.suptitle(f"Time step: {t}")
        return ims

    anim = FuncAnimation(
        ims[0].figure, update, frames=t_range, blit=False, repeat=False
    )

    if save_path:
        anim.save(save_path, fps=fps)
        print(f"save to {save_path}")
    else:
        plt.show()


def pixel2voxel(data: np.ndarray, z_slices: int = 4) -> np.ndarray:
    """
    Expand (c, x, y, t) array to (c, x, y, z, t) by repeating along z-axis.

    Parameters:
    - data: np.ndarray, shape (c, x, y, t)
        Input data.
    - z_slices: int
        Number of slices along the z-axis.

    Returns:
    - np.ndarray, shape (c, x, y, z, t)
        Expanded data.
    """
    return data.unsqueeze(3).repeat(1, 1, 1, z_slices, 1)


def pred_plot3d(
    y: np.ndarray,
    p: np.ndarray,
    t: int,
    fig: Optional[plt.Figure] = None,
    show_colorbar: bool = True,
    vmins: Optional[List[float]] = None,
    vmaxs: Optional[List[float]] = None,
    plot_method: str = "contourf",
    level: int = 20,
    cmap: str = "viridis",
    shade: bool = False,
) -> np.ndarray:
    """
    2x3 subplot of 3D voxel data (x,y,z,t) from (c,x,y,t) expanded by z_slices.

    Parameters:
    - y: np.ndarray, shape (3, height, width, time)
        Ground truth data.
    - p: np.ndarray, shape (3, height, width, time)
        Predicted data.
    - t: int
        Time step to visualize.
    - fig: Optional[plt.Figure]
        Matplotlib figure object. If None, a new figure is created.
    - show_colorbar: bool
        Whether to display colorbars for each subplot.
    - vmins: Optional[List[float]]
        Minimum values for color normalization. If None, computed from data.
    - vmaxs: Optional[List[float]]
        Maximum values for color normalization. If None, computed from data.
    - plot_method: str
        Plotting method, either 'image' or 'contourf'.
    - level: int
        Number of contour levels for 'contourf' method.
    - cmap: str
        Colormap to use for plotting.
    - shade: bool
        Whether to apply shading to 3D plots.

    Returns:
    - np.ndarray
        Axes used for the plots.
    """
    truth = pixel2voxel(y)
    pred = pixel2voxel(p)

    if fig is None:
        fig = plt.figure(figsize=(7, 5))
    else:
        fig.clf()
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.05])
    axes = np.empty((2, 3), dtype=object)

    if vmins is None:
        vmins = [
            min(truth[i, :, :, :, t].min(), pred[i, :, :, :, t].min()) for i in range(3)
        ]
    if vmaxs is None:
        vmaxs = [
            max(truth[i, :, :, :, t].max(), pred[i, :, :, :, t].max()) for i in range(3)
        ]

    titles = ["Electric Field", "Deformation Field", "Damage Indicator"]
    labels = [r"$\xi_3$ (V/m)", r"$F_{22}$", r"$I_D$"]
    norms = [mcolors.Normalize(vmin=vmins[i], vmax=vmaxs[i]) for i in range(3)]

    for row, data3d in enumerate([truth, pred]):
        for i in range(3):
            ax = fig.add_subplot(gs[row, i], projection="3d")
            axes[row, i] = ax
            vol = data3d[i, :, :, :, t].numpy().copy()
            x_len, y_len, z_len = vol.shape
            norm = norms[i]

            # generate mesh
            X, Y, Z = np.meshgrid(np.arange(x_len), np.arange(y_len), np.arange(z_len))
            xmin, xmax = X.min(), X.max()
            ymin, ymax = Y.min(), Y.max()
            zmin, zmax = Z.min(), Z.max()

            # plot
            if plot_method == "image":
                cmap = plt.get_cmap(cmap)

                # plot image
                top_color = cmap(norms[i](vol[:, :, -1]))
                ax.plot_surface(
                    X[:, :, -1],
                    Y[:, :, -1],
                    Z[:, :, -1],
                    rstride=1,
                    cstride=1,
                    facecolors=top_color,
                    shade=shade,
                )

                front_color = cmap(norms[i](vol[:, -1, :]))
                ax.plot_surface(
                    X[:, -1, :],
                    Y[:, -1, :],
                    Z[:, -1, :],
                    rstride=1,
                    cstride=1,
                    facecolors=front_color,
                    shade=shade,
                )

                side_color = cmap(norms[i](vol[-1, :, :]))
                ax.plot_surface(
                    X[-1, :, :],
                    Y[-1, :, :],
                    Z[-1, :, :],
                    rstride=1,
                    cstride=1,
                    facecolors=side_color,
                    shade=shade,
                )

            elif plot_method == "contourf":
                levels = np.linspace(vmins[i], vmaxs[i], level)
                kw = dict(levels=levels, norm=norm, extend="both", cmap=cmap)

                # plot contourf surfaces
                ax.contourf(
                    X[:, :, -1],
                    Y[:, :, -1],
                    vol[:, :, -1],
                    zdir="z",
                    offset=Z.max(),
                    **kw,
                )
                ax.contourf(
                    X[-1, :, :],
                    vol[-1, :, :],
                    Z[-1, :, :],
                    zdir="y",
                    offset=Y.max(),
                    **kw,
                )
                ax.contourf(
                    vol[:, -1, :],
                    Y[:, -1, :],
                    Z[:, -1, :],
                    zdir="x",
                    offset=X.max(),
                    **kw,
                )

            else:
                raise ValueError("plot_method must be 'image' or 'contourf'")

            # plot edges
            edges_kw = dict(color="k", lw=0.25, zorder=1e3)
            ax.plot([xmax, xmax], [ymin, ymax], zmin, **edges_kw)
            ax.plot([xmax, xmax], [ymin, ymax], zmax, **edges_kw)
            ax.plot([xmin, xmin], [ymin, ymax], zmax, **edges_kw)
            ax.plot([xmin, xmax], [ymin, ymin], zmax, **edges_kw)
            ax.plot([xmin, xmax], [ymax, ymax], zmax, **edges_kw)
            ax.plot([xmin, xmax], [ymax, ymax], zmin, **edges_kw)
            ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
            ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)
            ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **edges_kw)

            # figure settings
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))
            ax.set_title(titles[i] if row == 0 else "")
            ax.set_axis_off()
            ax.set_box_aspect([1 / 8, 1, 1])
            ax.view_init(azim=15, vertical_axis="y")

    # Colorbar
    if show_colorbar:
        for i in range(3):
            cax = fig.add_subplot(gs[2, i])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norms[i])
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cbar.set_label(labels[i])
            cbar.set_ticks([vmins[i], vmaxs[i]])

    fig.text(0.1, 0.73, "Ground truth", va="center", ha="right", rotation=90)
    fig.text(0.1, 0.32, "Prediction", va="center", ha="right", rotation=90)
    return axes


def pred_anim3d(
    y: np.ndarray,
    p: np.ndarray,
    t_range: Optional[range] = None,
    save_path: Optional[str] = None,
    fps: int = 2,
    **kwargs,
) -> None:
    """
    Create a 3D animation of ground truth and predictions over a range of time steps.

    Parameters:
    - y: np.ndarray, shape (3, height, width, time)
        Ground truth data.
    - p: np.ndarray, shape (3, height, width, time)
        Predicted data.
    - t_range: Optional[range]
        Range of time steps to animate.
    - save_path: Optional[str]
        Path to save the animation. If None, the animation is displayed.
    - fps: int
        Frames per second for the animation.
    - kwargs: dict
        Additional keyword arguments for `pred_plot3d` function.
    """
    if t_range is None:
        t_range = range(y.shape[-1])

    fig = plt.figure(figsize=(7, 5))

    def update(t):
        for ax in fig.axes:
            ax.clear()
        pred_plot3d(y, p, t, fig=fig, **kwargs)
        fig.suptitle(f"Time step: {t}")
        return fig.axes

    anim = FuncAnimation(fig, update, frames=t_range, blit=False, repeat=False)

    if save_path:
        anim.save(save_path, fps=fps, dpi=300)
        print(f"save to {save_path}")
    else:
        plt.show()
