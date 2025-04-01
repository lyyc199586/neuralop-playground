import numpy as np
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
          index: y[channel, height, width, time]
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
        im = ax.imshow(y[i, :, :, t], origin='lower', norm=norms[i])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    # Prediction
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        axes[1, i] = ax
        im = ax.imshow(pred[i, :, :, t], origin='lower', norm=norms[i])
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