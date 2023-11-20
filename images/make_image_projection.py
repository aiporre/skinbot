import glob

import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
def plot_proj(fname):
    C = image.imread(fname)

    xp, yp, __ = C.shape

    x = np.arange(0, xp, 1)
    y = np.arange(0, yp, 1)
    Y, X = np.meshgrid(y, x)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d')
    ax.dist=6.2
    ax.view_init(elev=45, azim=-45)

    # ax.plot_surface(X, Y, X-X+yp, facecolors=C,
    #                 rstride=2, cstride=2,
    #                 antialiased=True, shade=False)
    #
    # ax.plot_surface(X, X-X, Y, facecolors=np.fliplr(C.transpose((1,0,2))),
    #                 rstride=2, cstride=2,
    #                 antialiased=True, shade=False)
    #
    # ax.plot_surface(X-X+xp, X, Y, facecolors=np.fliplr(C.transpose((1,0,2))),
    #                 rstride=2, cstride=2,
    #                 antialiased=True, shade=False)

    # ax.plot_surface(X, X-X, Y, facecolors=C.transpose((1,0,2)),
    #                 rstride=2, cstride=2,
    #                 antialiased=True, shade=False)

    ax.plot_surface(X-X+xp, X, Y, facecolors=C,
                    rstride=2, cstride=2,
                    antialiased=True, shade=False)
    # turn off the axis planes
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.set_axis_off()
    fname_base = fname.split('.')[0]
    plt.savefig(f'{fname_base}_proj.png', dpi=300, transparent=True)

for f in glob.glob('*.png'):
    if '_proj.png' not in f:
        plot_proj(f)
        print(f)
# plt.show()