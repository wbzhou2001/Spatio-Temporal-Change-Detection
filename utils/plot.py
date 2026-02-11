import matplotlib.pyplot as plt
import numpy as np

def plot_our_result(ax, center, radius, nres=100, **kwds):
    '''
    Args:
    - center: [ ncenter, 2 ] np
    - radius: scalar
    - nres: resolution of plotting
    '''
    # Create binary mask
    mask = np.zeros((nres, nres))
    
    for i in range(nres):
        for j in range(nres):
            x = (i + 0.5) / nres
            y = (j + 0.5) / nres
            c = np.array([x, y]) # [2] np
            dist = np.linalg.norm(center - c[None, :], ord=np.inf, axis=1)  # [ ncenter ] np
            if np.any(dist <= radius):
                mask[j, i] = 1
    
    # Draw boundary only
    kwds_ = kwds.copy()
    kwds_.pop('label', None)
    ax.contour(mask, levels=[0.5], extent=(0, 1, 0, 1), origin='lower', **kwds_)

    # Add label if provided
    if 'label' in kwds:
        rect_kwds = {k: v for k, v in kwds.items() if k in ['edgecolor', 'facecolor', 'linestyle', 'alpha']}
        ax.add_patch(plt.Rectangle((c[0], c[1]), 0, 0,
                                   color=kwds['colors'],
                                   fill=False,
                                   linewidth=kwds['linewidths'],
                                   label=kwds['label']), **rect_kwds)

        # ax.add_patch(plt.Rectangle((c[0], c[1]), 0, 0, **kwds))

    return ax

def visualize(center_list, radius, data, center_post, radius_post, save_path=None):

    fig_scale = 0.8
    fig, axes = plt.subplots(1, 4, figsize=(15*fig_scale,3*fig_scale))

    for i in range(4):
        ax = axes[i]
        # events
        mask = data[:, 0] <= (i + 1) / 4
        ax.scatter(data[mask, 1], data[mask, 2], color='lightgray', s=10)

        # our results
        plt_kwds = {
            'colors': 'red',
            'linewidths': 3,
            'label': r'\textbf{Ours}'
        }
        plot_our_result(ax, center_list[(i+1)*100//4-1], radius, **plt_kwds)
        ax.set_title(rf'$t={((i+1)/4):.2f}$')

        if i != 0:
            # true change region
            plt_kwds = {
                'colors': 'black',
                'linewidths': 3,
                'label': r'True Change Region $\Omega$'
            }
            plot_our_result(ax, center_post, radius_post, **plt_kwds)

    axes[-1].legend(loc='center', bbox_to_anchor=(-1.65*fig_scale, -0.3*fig_scale), ncol=5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()