import numpy as np
import matplotlib
# matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

def data_image(x, y, t, axes=None, show_colorbar=True):
    """
    2*3 subplots
    plot  input x: x[0], x[1], x[2]
          output y: y[0], y[1], y[2]
    index: x[channel, height, width, time]
    """
    
    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    else:
        fig = axes[0, 0].figure
    
    # plot x: Ek(x,t); M(x); RAMPs(x)
    axes[0, 0].annotate(f"x", xy=(-0.15, 0.5), 
             xycoords="axes fraction",
             ha="right", va="center", rotation=90)
    
    Ek = np.ma.masked_where(x[0,:,:,t]==0, x[0,:,:,t])
    viridis_white = plt.cm.viridis.copy()
    viridis_white.set_bad(color='white')
    
    im0 = axes[0, 0].imshow(Ek, cmap=viridis_white, origin='lower')
    axes[0, 0].set_title('Sensor signal $E_k(x, t)$')
    
    im1 = axes[0, 1].imshow(x[1,:,:,t], cmap='gray_r', origin='lower')
    axes[0, 1].set_title('Mask $M(x)$')
    
    im2 = axes[0, 2].imshow(x[2,:,:,t], origin='lower')
    axes[0, 2].set_title('RAMPs field $\Phi(x)$')
    
    # plot y: E(x, t); F(x, t); D(x, t)
    axes[1, 0].annotate(f"y", xy=(-0.15, 0.5), 
             xycoords="axes fraction",
             ha="right", va="center", rotation=90)
    im3 = axes[1, 0].imshow(y[0,:,:,t], origin='lower')
    axes[1, 0].set_title('Electricity $E(x, t)$')
    
    im4 = axes[1, 1].imshow(y[1,:,:,t], origin='lower')
    axes[1, 1].set_title('Defgrad $F(x, t)$')
    
    im5 = axes[1, 2].imshow(y[2,:,:,t], origin='lower')
    axes[1, 2].set_title('Damage $D(x, t)$')
    
    # clean ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    # set colorbar
    if show_colorbar:
        ims = [im0, im1, im2, im3, im4, im5]
        for ax, im in zip(axes.flat, ims):
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.07, location='bottom')

    
    plt.tight_layout()
    plt.show()
    
    return axes

def pred_plot(y, pred, t, show_colorbar=True, vmins=None, vmaxs=None, plot_method='image'):
    """
    2*3 subplots
    plot: ground-truth y: y[0], y[1], y[2]
          prediction: pred[0], pred[1], pred[2]
    index: [channel, height, width, time]
    plot_method: 'image' or 'contourf'
    """
    
    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.05])
    axes = np.empty((2, 3), dtype=object)
        
    if vmins is None:
        vmins = [min(y[i, :, :, t].min(), pred[i, :, :, t].min()) for i in range(3)]
    if vmaxs is None:
        vmaxs = [max(y[i, :, :, t].max(), pred[i, :, :, t].max()) for i in range(3)]
    
    titles = ['Electric Field', 'Deformation Field', 'Damage Indicator']
    labels = [r'$\xi_3$ (V/m)', r'$F_{22}$', r'$I_D$']
    ims = []
    norms = [mcolors.Normalize(vmin=vmins[i], vmax=vmaxs[i]) for i in range(3)]

    # Ground truth
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        axes[0, i] = ax
        data = y[i, :, :, t]
        if plot_method == 'contourf':
            levels = np.linspace(vmins[i], vmaxs[i], 21)
            im = ax.contourf(data, levels=levels, extend='both')
        else:
            im = ax.imshow(data, origin='lower', norm=norms[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    # Prediction
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        axes[1, i] = ax
        data = pred[i, :, :, t]
        if plot_method == 'contourf':
            levels = np.linspace(vmins[i], vmaxs[i], 21)
            im = ax.contourf(data, levels=levels, extend='both') 
        else:
            im = ax.imshow(data, origin='lower', norm=norms[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    # colorbars
    if show_colorbar:
        for i in range(3):
            cax = fig.add_subplot(gs[2, i])
            cbar = fig.colorbar(ims[i+3], cax=cax, orientation='horizontal')
            cbar.set_label(labels[i])
            cbar.set_ticks([vmins[i], vmaxs[i]])
            
    fig.text(0.1, 0.73, 'Ground truth', va='center', ha='right', rotation=90)
    fig.text(0.1, 0.32, 'Prediction', va='center', ha='right', rotation=90)
    return axes, ims

def pred_anim(y, pred, t_range, save_path=None, fps=2, **kwargs):
    
    if t_range is None:
        t_range = range(y.shape[-1])
    
    _, ims = pred_plot(y, pred, t_range[0], **kwargs)
    def update(t):
        for i in range(3):
            ims[i].set_array(y[i, :, :, t])
            ims[i+3].set_array(pred[i, :, :, t])
        return ims
    
    anim = FuncAnimation(ims[0].figure, update, frames=t_range,  blit=False, repeat=False)
    # plt.tight_layout()
    
    if save_path:
        anim.save(save_path, fps=fps)
        print(f"save to {save_path}")
    else:
        plt.show()
        
def pixel2voxel(data, z_slices=4):
    """
    Expand (c, x, y, t) array to (c, x, y, z, t) by repeating along z-axis.
    """
    return data.unsqueeze(3).repeat(1, 1, 1, z_slices, 1)
        
def pred_voxel(y, p, t, fig=None, show_colorbar=True, vmins=None, vmaxs=None, plot_method='contourf'):
    """
    2x3 subplot of 3D voxel data (x,y,z,t) from (c,x,y,t) expanded by z_slices.

    Parameters:
    - y, p: ndarray of shape (c, x, y, t)
    - t: int, time step
    - plot_method: 'voxel' or 'contourf'
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
        vmins = [min(truth[i, :, :, :, t].min(), pred[i, :, :, :, t].min()) for i in range(3)]
    if vmaxs is None:
        vmaxs = [max(truth[i, :, :, :, t].max(), pred[i, :, :, :, t].max()) for i in range(3)]

    titles = ['Electric Field', 'Deformation Field', 'Damage Indicator']
    labels = [r'$\xi_3$ (V/m)', r'$F_{22}$', r'$I_D$']
    norms = [mcolors.Normalize(vmin=vmins[i], vmax=vmaxs[i]) for i in range(3)]
    
    for row, data3d in enumerate([truth, pred]):
        for i in range(3):
            ax = fig.add_subplot(gs[row, i], projection='3d')
            axes[row, i] = ax
            vol = data3d[i, :, :, :, t].numpy().copy()
            x_len, y_len, z_len = vol.shape
            norm = norms[i]
            X, Y, Z = np.meshgrid(np.arange(x_len), np.arange(y_len), np.arange(z_len))
            xmin, xmax = X.min(), X.max()
            ymin, ymax = Y.min(), Y.max()
            zmin, zmax = Z.min(), Z.max()
            if plot_method == 'voxel':
                ls = mcolors.LightSource(azdeg=120, altdeg=45)
                facecolors = plt.cm.viridis(norm(vol))
                ax.voxels(vol, facecolors=facecolors, edgecolor='w', shade=True, lightsource=ls, lw=0)
            elif plot_method == 'contourf':
                levels = np.linspace(vmins[i], vmaxs[i], 50)
                kw = dict(levels=levels, norm=norm, extend='both')

                # plot contourf surfaces
                ax.contourf(X[:, :, -1], Y[:, :, -1], vol[:, :, -1], zdir="z", offset=Z.max(), **kw)
                ax.contourf(X[-1, :, :], vol[-1, :, :], Z[-1, :, :], zdir="y", offset=Y.max(), **kw)
                ax.contourf(vol[:, -1, :], Y[:, -1, :], Z[:, -1, :], zdir="x",offset=X.max(), **kw)

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
                
                ax.set(xlim=(xmin, xmax),
                       ylim=(ymin, ymax),
                       zlim=(zmin, zmax))
            else:
                raise ValueError("plot_method must be 'voxel' or 'contourf'")

            ax.set_title(titles[i] if row == 0 else "")
            ax.set_axis_off()
            ax.set_box_aspect([1/8, 1, 1])
            ax.view_init(azim=15, vertical_axis='y')
    
    # Colorbar
    if show_colorbar:
        for i in range(3):
            cax = fig.add_subplot(gs[2, i])
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norms[i])
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
            cbar.set_label(labels[i])
            cbar.set_ticks([vmins[i], vmaxs[i]])
            
    fig.text(0.1, 0.73, 'Ground truth', va='center', ha='right', rotation=90)
    fig.text(0.1, 0.32, 'Prediction', va='center', ha='right', rotation=90)
    return axes

def pred_voxel_anim(y, p, t_range, save_path=None, fps=2, **kwargs):
    
    if t_range is None:
        t_range = range(y.shape[-1])
    
    fig = plt.figure(figsize=(7, 5))
    
    def update(t):
        for ax in fig.axes:
            ax.clear()
        pred_voxel(y, p, t, fig=fig, **kwargs)
        fig.suptitle(f"Time step: {t}")
        return fig.axes
    
    anim = FuncAnimation(fig, update, frames=t_range,  blit=False, repeat=False)
    
    if save_path:
        anim.save(save_path, fps=fps)
        print(f"save to {save_path}")
    else:
        plt.show()