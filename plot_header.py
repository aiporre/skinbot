import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

root  = "/home/sauron/Documents/Phd/skin_project/images_for_presentation/selected_images_samples"
# search for "JPG" and "jpg" and "png" "PNG" files
images_all = []
for root, dirs, files in os.walk(root):
    for file in files:
        if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".png") or file.endswith(".PNG"):
            print(os.path.join(root, file))
            images_all.append(os.path.join(root, file))
# select mask
mask = {}
for image in images_all:
    if "mask" in image:
        label = image.split("/")[-1].split("_")[0].lower()
        mask[label] = image
images = {}
for image in images_all:
    if "mask" not in image:
        label = image.split("/")[-1].split("_")[0].lower()
        images[label] = image
print(len(images))
print(len(mask))
# plot a 2x7 grid
fig, axs = plt.subplots(1, 7, figsize=(20, 10), gridspec_kw = {'wspace':0, 'hspace':0}, squeeze=True)
# plot top row images and bottom row masks
for i, k in enumerate(images.keys()):
    img = mpimg.imread(images[k])
    # if vaskulitis or infection rotate the image 90 degrees
    if k == "vaskulitis" or k == "infection":
        img = np.rot90(img)
    axs[i].imshow(img)
    axs[i].set_title(k)
    axs[i].axis('off')
    # mask_img = mpimg.imread(mask[k])
    # if k == "vaskulitis" or k == "infection":
    #     mask_img = np.rot90(mask_img)
    # axs[1, i].imshow(mask_img)
    # axs[1, i].set_title(k)
    # axs[1, i].axis('off')
plt.show()
