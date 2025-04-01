import numpy as np
import matplotlib.pyplot as plt

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

def pred_image(x, y, pred, t, axes=None, show_colorbar=True):
    """
    3*3 subplots
    plot  input x: x[0], x[1], x[2]
          output y: y[0], y[1], y[2]
          pred: pred[0], pred[1], pred[2]
    index: x[channel, height, width, time]
    """
    
    if axes is None:
        fig, axes = plt.subplots(3, 3, figsize=(7, 7.5))
    else:
        fig = axes[0, 0].figure
    
    # plot x: Ek(x,t); M(x); RAMPs(x)
    axes[0, 0].annotate(f"Input", xy=(-0.15, 0.5), 
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
    axes[1, 0].annotate(f"Ground truth", xy=(-0.15, 0.5), 
             xycoords="axes fraction",
             ha="right", va="center", rotation=90)
    
    im3 = axes[1, 0].imshow(y[0,:,:,t], origin='lower')
    axes[1, 0].set_title('Electricity $E(x, t)$')
    
    im4 = axes[1, 1].imshow(y[1,:,:,t], origin='lower')
    axes[1, 1].set_title('Defgrad $F(x, t)$')
    
    im5 = axes[1, 2].imshow(y[2,:,:,t], origin='lower')
    axes[1, 2].set_title('Damage $D(x, t)$')
    
    # plot pred: E(x, t); F(x, t); D(x, t)
    axes[2, 0].annotate(f"Prediction", xy=(-0.15, 0.5), 
             xycoords="axes fraction",
             ha="right", va="center", rotation=90)
    
    im6 = axes[2, 0].imshow(pred[0,:,:,t], origin='lower')
    axes[2, 0].set_title('Electricity $E(x, t)$')
    
    im7 = axes[2, 1].imshow(pred[1,:,:,t], origin='lower')
    axes[2, 1].set_title('Defgrad $F(x, t)$')
    
    im8 = axes[2, 2].imshow(pred[2,:,:,t], origin='lower')
    axes[2, 2].set_title('Damage $D(x, t)$')
    
    # clean ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    # set colorbar
    if show_colorbar:
        ims = [im0, im1, im2, im3, im4, im5, im6, im7, im8]
        for ax, im in zip(axes.flat, ims):
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.07, location='bottom')

    
    plt.tight_layout()
    plt.show()
    
    return axes