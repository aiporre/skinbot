import numpy as np
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd


def get_dominant_color(image):
    NUMBER_OF_CLUSTERS = 3
    r = []
    g = []
    b = []
#     batman_image = 255*(batman_image-batman_image.min())/(batman_image.max()-batman_image.min())
    image = image.transpose(0, 2).transpose(0, 1).numpy()
    M,N,C = image.shape
    image.reshape(M * N, C)
    for row in image:
        for temp_r, temp_g, temp_b in row:
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)

    batman_df = pd.DataFrame({'red' : r,
                            'green' : g,
                            'blue' : b})
    batman_df['scaled_color_red'] = whiten(batman_df['red'])
    batman_df['scaled_color_blue'] = whiten(batman_df['blue'])
    batman_df['scaled_color_green'] = whiten(batman_df['green'])

    cluster_centers, _ = kmeans(batman_df[['scaled_color_red',
                                        'scaled_color_blue',
                                        'scaled_color_green']], NUMBER_OF_CLUSTERS)

    dominant_colors = []

#     red_std, green_std, blue_std = batman_df[['red',
#                                             'green',
#                                             'blue']].std()
#     print('number of clusters= ', len(cluster_centers))

    for cluster_center in cluster_centers:
        cluster_center = (cluster_center - cluster_center.min())/(cluster_center.max()-cluster_center.min())
        red_scaled, green_scaled, blue_scaled = cluster_center
        dominant_colors.append((
            red_scaled,
            green_scaled,
            blue_scaled
        ))
    if len(dominant_colors) == 1:
        dominant_colors.append(dominant_colors[0])
        dominant_colors.append(dominant_colors[0])
    if len(dominant_colors) == 2:
        aux = (np.array(dominant_colors[0]) + np.array(dominant_colors[1]))/2
        dominant_colors.append(aux)
    return dominant_colors