import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt

C = image.imread('heatmap.png')

xp, yp, __ = C.shape

x = np.arange(0, xp, 1)
y = np.arange(0, yp, 1)
Y, X = np.meshgrid(y, x)

fig = plt.figure(figsize=(12,9))
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
plt.savefig('heatmap_proj.png', dpi=300, transparent=True)
# plt.show()