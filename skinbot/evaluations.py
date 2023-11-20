import skinbot.skinlogging as logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from skinbot.autoencoders import AutoEncoder, VariationalAutoEncoder, ConvolutionalAutoEncoder, AutoEncoderClassifier
import torchextractor as tx
from skinbot.config import Config
from skinbot.dataset import crop_lesion, read_image
import numpy as np
from skimage.transform import resize
C = Config()


def plot_one_grad_cam(model, dataloader, target_mode= "single", fname=None, index=0, target_layer="layer4.2.conv3"):
    """
    plots one calculation of grad cam on model by fiename or index
    """
    dataset = dataloader.dataset
    if fname is not None:
        logging.info('Fname overwrites index be aware')
        index = dataset.image_fnames.index(fname)
    else:
        fname = dataset.image_fnames[index]
    logging.info(f"Plotting gradCAM for image: {fname} at index {index}")
    # TODO: use variable targe_later +layer3.2.conv3
    target_layers_objs = [model.layer4[-1]]

    sample = dataset[index]
    x,y = sample
    x = torch.unsqueeze(x,0)
    use_cuda = torch.has_cuda
    cam = GradCAM(model=model, target_layers=target_layers_objs, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(y)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=x, targets=targets)

    # fname_19 = os.path.join(dataset.images_dir, dataset.image_fnames[index])
    # if "crop" in target_mode:  # crop in target:
    #     rgb_img = read_image(fname_19).numpy() #.transpose(1, 2, 0)
    # else:
    #     rgb_img = read_image(fname_19).numpy()
    #     cc = dataset._crop_boxes[dataset.image_fnames[index]]
    #     rgb_img = crop_lesion(rgb_img, cc).transpose(1, 2, 0)
    # In this example grayscale_cam has only one image in the batch:
    # rgb_img = rgb_img/255.0
    rgb_img = (x - x.min())/(x.max()-x.min())
    rgb_img = rgb_img.numpy().squeeze().transpose(1,2,0)
    print(rgb_img.shape)
    # plt.imshow(rgb_img)
    # plt.show()
    fig1, ax = plt.subplots(ncols=2)
    grayscale_cam = grayscale_cam[0, :]
    # resize to original shape of image
    # shape = list(dataset.get_image_shape(index))
    # rgb_img = resize(rgb_img, shape)
    # grayscale_cam = resize(grayscale_cam, shape[1:])
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.6)
    # visualization = resize(visualization, output_shape=dataset.get_image_shape(index)).transpose(1, 2, 0)
    ax[0].imshow(rgb_img)
    ax[0].axis('off')
    ax[0].set_title(dataset.image_fnames[index])
    ax[1].imshow(visualization)
    ax[1].axis('off')
    return fig1, ax

def plot_classification_features(model, dataloader, target_mode= "single", fname=None, index=0, target_layer="layer4.2.conv3",
                                 device=None, dir='./feature_images', plot_image_next=False):
    """
    plots one calculation of grad cam on model by fiename or index
    """
    assert device is not None, 'device must be specified'

    dataset = dataloader.dataset
    if fname is not None:
        logging.info('the fname will overwrites index while plotting the classification features be aware')
        index = dataset.image_fnames.index(fname)
    else:
        fname = dataset.image_fnames[index]
    logging.info(f"Plotting gradCAM for image: {fname} at index {index}")
    # TODO: use variable targe_later +layer3.2.conv3
    target_layers_objs = [model.layer4[-1]]

    sample = dataset[index]
    x,y = sample
    x = torch.unsqueeze(x,0)
    x = x.to(device)
    # y = y.to(device)
    if isinstance(target_layer, str):
        target_layers = [target_layer]
    elif isinstance(target_layer, list):
        target_layers = target_layer
    else:
        raise ValueError(f'target_layer must be str or list of str, got {type(target_layer)}')

    model = tx.Extractor(model, target_layers)
    model.eval()
    pred, features = model(x)
    feature_shapes = {name: f.shape for name, f in features.items()}

    print(feature_shapes)
    rgb_img = (x - x.min())/(x.max()-x.min())
    rgb_img = rgb_img.cpu().numpy().squeeze().transpose(1,2,0)
    print(rgb_img.shape)
    # plot all channels and save in images in directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    for name, f in features.items():
        name = name.replace('.', '_')
        fname = fname.replace('.jpg', '')
        fname = fname.replace('.JPG', '')
        fig_shape = min(16, f.shape[1])
        for i in range(fig_shape):
            image_fname = os.path.join(dir, f'{fname}_{name}_dim{i}.png')
            print(image_fname)
            if plot_image_next:
                fig, ax = plt.subplots(1, 2, figsize=(5, 5))  # figsize(x,y)
                feature_image = f[0, i].cpu().detach().numpy()
                # feature_image = (feature_image - feature_image.min())/(feature_image.max()-feature_image.min())
                ax[0].imshow(rgb_img)
                ax[1].imshow(feature_image)
                # layout tight and axis off
                ax[0].axis('off')
                ax[1].axis('off')
            else:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                feature_image = f[0, i].cpu().detach().numpy()
                ax.imshow(feature_image)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(image_fname)
        # plt.savefig(image_fname)
        # plt.show()


    use_cuda = torch.has_cuda
    # cam = GradCAM(model=model, target_layers=target_layers_objs, use_cuda=use_cuda)
    # targets = [ClassifierOutputTarget(y)]
    #
    # # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=x, targets=targets)
    #
    # # fname_19 = os.path.join(dataset.images_dir, dataset.image_fnames[index])
    # # if "crop" in target_mode:  # crop in target:
    # #     rgb_img = read_image(fname_19).numpy() #.transpose(1, 2, 0)
    # # else:
    # #     rgb_img = read_image(fname_19).numpy()
    # #     cc = dataset._crop_boxes[dataset.image_fnames[index]]
    # #     rgb_img = crop_lesion(rgb_img, cc).transpose(1, 2, 0)
    # # In this example grayscale_cam has only one image in the batch:
    # # rgb_img = rgb_img/255.0

    # plt.imshow(rgb_img)
    # plt.show()
    # pred = model(rgb_img)
    # fig1, ax = plt.subplots(ncols=2)
    # grayscale_cam = grayscale_cam[0, :]
    # # resize to original shape of image
    # # shape = list(dataset.get_image_shape(index))
    # # rgb_img = resize(rgb_img, shape)
    # # grayscale_cam = resize(grayscale_cam, shape[1:])
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.6)
    # # visualization = resize(visualization, output_shape=dataset.get_image_shape(index)).transpose(1, 2, 0)
    # ax[0].imshow(rgb_img)
    # ax[0].axis('off')
    # ax[0].set_title(dataset.image_fnames[index])
    # ax[1].imshow(visualization)
    # ax[1].axis('off')
    # return fig1, ax


def predict_samples(model, dataloader, fold, target_mode, N=None, device=None):
    model.eval()
    labels = []
    predictions = []
    probs = []
    is_fuzzy = target_mode in ["fuzzy", "multi"]
    labels_fuzzy = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            inputs = inputs if device is None else inputs.to(device)
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.tolist())
            # if fuzzy we need to compute the index as the max value
            if is_fuzzy:
                labels_fuzzy.extend(label.tolist())
                label = torch.argmax(label, dim=1) if len(label.shape) > 1 else label
            labels.extend(label.tolist())
            probs.extend(prob.tolist())
            cnt = len(labels)
            if N is not None:
                logging.info('pred: ', cnt, 'out of ', N)
                if cnt > N:
                    break
    if is_fuzzy:
        return pd.DataFrame( {"y_pred": predictions, "y_true": labels, "y_prob": probs, "y_true_fuzzy": labels_fuzzy,
                              'files': dataloader.dataset.image_fnames,
                              'fold': len(predictions)*[fold]})
    else:
        return pd.DataFrame( {"y_pred": predictions, "y_true": labels, "y_prob": probs})

def error_analysis(df):
    df = df.copy()
    df["error"] = df["y_pred"] != df["y_true"]
    df["error"] = df["error"].astype(int)
    return df

def plot_bargraph_with_groupings(df, groupby, colourby, title, xlabel, ylabel, FIG_SIZE=(20,20)):
    """
    Plots a dataframe showing the frequency of datapoints grouped by one column and coloured by another.
    df : dataframe
    groupby: the column to groupby
    colourby: the column to color by
    title: the graph title
    xlabel: the x label,
    ylabel: the y label
    """
    # Makes a mapping from the unique colourby column items to a random color.
    ind_col_map = {x:y for x, y in zip(df[colourby].unique(),
                               [plt.cm.Paired(np.arange(len(df[colourby].unique())))][0])}


    # Find when the indicies of the soon to be bar graphs colors.
    unique_comb = df[[groupby, colourby]].drop_duplicates()
    name_ind_map = {x:y for x, y in zip(unique_comb[groupby], unique_comb[colourby])}
    c = df[groupby].value_counts().index.map(lambda x: ind_col_map[name_ind_map[x]])

    # Makes the bargraph.
    ax = df[groupby].value_counts().plot(kind='bar',
                                         figsize=FIG_SIZE,
                                         title=title,
                                         color=[c.values])
    # Makes a legend using the ind_col_map
    legend_list = []
    for key in ind_col_map.keys():
        legend_list.append(mpatches.Patch(color=ind_col_map[key], label=key))

    # display the graph.
    plt.legend(handles=legend_list)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_latent_space(autoencoder, num_classes, data_loader, device, save=False, dim_red=None):
    import matplotlib.colors as mcolors
    d = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)
            targets_num = torch.argmax(targets, dim=1)
            if isinstance(autoencoder, AutoEncoder):
                embedding = autoencoder.latent(features)
            elif isinstance(autoencoder, VariationalAutoEncoder):
                embedding = autoencoder.latent(features)
            elif isinstance(autoencoder, ConvolutionalAutoEncoder):
                if autoencoder.is_variational:
                    embedding = autoencoder.latent(features)
                else:
                    embedding = autoencoder.latent(features)
            elif isinstance(autoencoder, AutoEncoderClassifier):
                if autoencoder.is_variational:
                    embedding = autoencoder.latent(features)
                else:
                    embedding = autoencoder.latent(features)
            else:
                raise ValueError(f'autoencoder {type(autoencoder)} type not supported')

            for i in range(num_classes):
                if i in targets_num:
                    # print('torch.argmax(targets, dim=1)', torch.argmax(targets, dim=1))
                    mask = targets_num == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    fig, ax = plt.subplots(figsize=(5,5))
    labels_strings = list(C.labels.target_str_to_num.keys())
    if dim_red == 'tsne':
        from sklearn.manifold import TSNE
        X = []
        lenghts = []
        for i in range(num_classes):
            d[i] = np.concatenate(d[i])
            X.append(d[i])
            lenghts.append(d[i].shape[0])
        X = np.concatenate(X)

        X_embedded = TSNE(n_components=2, learning_rate='auto',
                          init='random', perplexity=3).fit_transform(X)
        ptr = 0
        for i in range(num_classes):
            d[i] = X_embedded[ptr:ptr+lenghts[i]]
            ptr += lenghts[i]
            ax.scatter(
                d[i][:, 0], d[i][:, 1],
                color=colors[i][1],
                label=f'{labels_strings[i]}',
                alpha=0.5,
                s=3**2)

    else:
        for i in range(num_classes):
            print('i', i)
            d[i] = np.concatenate(d[i])
            ax.scatter(
                d[i][:, 0], d[i][:, 1],
                color=colors[i][1],
                label=f'{labels_strings[i]}',
                alpha=0.5,
                s=3**2)

    ax.legend()
    if save:
        fig.savefig('latent_space.png')
    return fig, ax


def plot_detection(image, pred, label, all_titles_labels, save=True, show=False, mask=False, suffix='', fname=''):
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    pred = pred[0]
    colors = list(mcolors.TABLEAU_COLORS.items())
    image = image.cpu().numpy().transpose(1, 2, 0)
    print('colors')
    print(colors[0])

    def hex_to_rgb(hex_code):
        hex_code = hex_code.strip('#')
        red, green, blue = int(hex_code[:2], 16), int(hex_code[2:4], 16), int(hex_code[4:], 16)
        return (red, green, blue)

    if not mask:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'{fname}')
        ax[0,0].imshow(image)
        # iterate over boxes in label
        boxes = label['boxes'].cpu().numpy()
        labels = label['labels'].cpu().numpy()
        num_to_str = {v:k for k, v in all_titles_labels.items()}
        for i, box in enumerate(boxes):
            # print('box', box)
            x1, y1, x2, y2 = box
            label_i = labels[i]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[label_i][1], facecolor='none')
            ax[0,0].add_patch(rect)
            # write the label inside the box
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/8.0
            text = ax[0,0].annotate(num_to_str[label_i], (cx, cy), color='w', weight='bold', ha='center', va='center', size=8)
            text.set_bbox(dict(facecolor=colors[label_i][1], alpha=0.5, edgecolor='black'))

        ax[0,0].set_title('inputs')
        # do the same for the prediction
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        ax[0,1].imshow(image)
        for i, box in enumerate(boxes):
            # print('box', box)
            x1, y1, x2, y2 = box
            label_i = labels[i]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[label_i][1], facecolor='none')
            ax[0,1].add_patch(rect)
            # write the label inside the box
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/8.0
            text = ax[0,1].annotate(num_to_str[label_i], (cx, cy), color='w', weight='bold', ha='center', va='center', size=8)
            text.set_bbox(dict(facecolor=colors[label_i][1], alpha=0.5, edgecolor='black'))
        ax[0,1].set_title('predictions')
    else:
        images = []
        for img in dataset[:16]:
            images.append(img[0].cpu().numpy().transpose(1, 2, 0)) #C,H,W, H.W,C
        # create plot grid  4x4
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                index = i*4+j
                ax[i,j].imshow(images[index])
                ax[i,j].axis('off')
        plt.show()

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        fig.suptitle(f'{fname}')
        ax[0,0].imshow(image)
        # iterate over boxes in label
        boxes = label['boxes'].cpu().numpy()
        labels = label['labels'].cpu().numpy()
        masks = label['masks'].cpu().numpy()
        num_to_str = {v:k for k, v in all_titles_labels.items()}
        for i, box in enumerate(boxes):
            # print('box', box)
            x1, y1, x2, y2 = box
            label_i = labels[i]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[label_i][1], facecolor='none')
            ax[0,0].add_patch(rect)
            # write the label inside the box
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/8.0
            text = ax[0,0].annotate(num_to_str[label_i], (cx, cy), color='w', weight='bold', ha='center', va='center', size=8)
            text.set_bbox(dict(facecolor=colors[label_i][1], alpha=0.5, edgecolor='black'))
        ax[0,0].set_title('inputs')
        # plot masks for the input
        ax[1,0].imshow(image)
        for i, mask in enumerate(masks):
            mask = (mask.squeeze()>0.5).astype('float')
            label_i = labels[i]
            color_vect = hex_to_rgb(colors[label_i][1])
            mask_color = np.ones((mask.shape[0], mask.shape[1], 4), dtype='uint8')
            mask_color[:,:,0] = mask*color_vect[0]
            mask_color[:,:,1] = mask*color_vect[1]
            mask_color[:,:,2] = mask*color_vect[2]
            mask_color[:,:,3] = (127*mask).astype('uint8')
            ax[1,0].imshow(mask_color)
            ax[1,0].set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax[1,0].set_title('masks')

        # do the same for the prediction
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        ax[0,1].imshow(image)
        for i, box in enumerate(boxes):
            # print('box', box)
            x1, y1, x2, y2 = box
            label_i = labels[i]
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=colors[label_i][1], facecolor='none')
            ax[0,1].add_patch(rect)
            # write the label inside the box
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width()/2.0
            cy = ry + rect.get_height()/8.0
            text = ax[0,1].annotate(num_to_str[label_i], (cx, cy), color='w', weight='bold', ha='center', va='center', size=8)
            text.set_bbox(dict(facecolor=colors[label_i][1], alpha=0.5, edgecolor='black'))
        ax[0,1].set_title('predictions')

        # plot masks for the input
        ax[1, 1].imshow(image)
        for i, mask in enumerate(masks):
            mask = (mask.squeeze() > 0.5).astype('float')
            label_i = labels[i]
            color_vect = hex_to_rgb(colors[label_i][1])
            mask_color = np.ones((mask.shape[0], mask.shape[1], 4), dtype='uint8')
            mask_color[:,:,0] = mask*color_vect[0]
            mask_color[:,:,1] = mask*color_vect[1]
            mask_color[:,:,2] = mask*color_vect[2]
            mask_color[:,:,3] = (127*mask).astype('uint8')
            ax[1,1].imshow(mask_color)
            ax[1,1].set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax[1, 1].set_title('pred masks')
    if save:
        fig.savefig(f'detection_{suffix}.png')
    return fig, ax
