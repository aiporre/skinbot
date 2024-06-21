import tqdm

import skinbot.skinlogging
import os
import random

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from skinbot.dataset import get_dataloaders
from skinbot.config import read_config, Config
from skinbot.engine import create_classification_trainer, configure_engines, create_detection_trainer, \
    create_classification_evaluator, create_detection_evaluator, get_best_iteration, create_segmentation_trainer, \
    create_segmentation_evaluator, create_autoencoder_trainer, create_autoencoder_evaluator, get_last_checkpoint
from skinbot.evaluations import predict_samples, error_analysis, plot_one_grad_cam, plot_latent_space, plot_detection, \
                                plot_classification_features
from skinbot.models import get_model
import skinbot.skinlogging as logging
# from skinbot.transformers import num_classes, target_str_to_num



from skinbot.utils import validate_target_mode, configure_logging

C = Config()

def main(best_or_last='best',
         target_mode='single',
         model_name='resnet101',
         fold=0,
         epochs=10,
         batch_size=32,
         lr=0.001,
         momentum=0.8,
         optimizer='SGD',
         freeze='No',
         device='cuda',
         only_eval=False,
         patience=None,
         model_path=None,
         external_data=False,
         config_file='config.ini',
         ae_model_path=None):
    # log_interval = 1
    config = read_config(config_file)
    C = Config()
    C.set_config(config)
    configure_logging(config)
    # root_dir = config["DATASET"]["root"]
    # best_or_last = 'best'
    # only_eval = False
    # fold = 0
    # model_name = 'resnet101'
    # fuzzy_labels = True
    EPOCHS = epochs
    LR = lr  # 0.001
    display_info = True
    # target_mode = "single"
    # gpu device
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    # prepare dataset
    assert validate_target_mode(target_mode, ['single', 'multiple', 'fuzzy','segmentation', 'detection',
                                              'reconstruction', 'classification'])
    if 'multiple' in target_mode.lower() or 'fuzzy' in target_mode.lower():
        assert config['DATASET']['labels'] == 'all', f'Target mode Multiple and fuzzy not compatible with labels in {config_file} use config[dataset][labels] = all'
    if 'detection' in target_mode:
        valid_detection_models = ['fasterresnet50', 'fastermobilenet','maskrcnn']
        assert model_name in valid_detection_models, f'{target_mode} requires model ({valid_detection_models})'

    if 'construction' in target_mode:
        valid_detection_models = ['ae', 'vae', 'cae', 'cvae', 'convae', 'convvae', 'convcvae']
        assert model_name.lower() in valid_detection_models, f'{target_mode} requires model ({valid_detection_models}). Not {model_name}'

    _fold = fold if not external_data else None
    test_dataloader = get_dataloaders(config, batch=batch_size, mode='test', fold_iteration=_fold, target=target_mode)
    train_dataloader = get_dataloaders(config, batch=batch_size, mode='train', fold_iteration=_fold, target=target_mode)

    # prepare models
    print('------------> ', ae_model_path==-1)
    if ae_model_path == -1:
        ae_model_path = get_best_iteration('best_models', fold, model_name='convcvae', target_mode='reconstruction')
        ae_model_path = os.path.join('best_models', ae_model_path)
        print('best_model_path for autoencoder: ', ae_model_path)

    model, optimizer = get_model(model_name, optimizer=optimizer, lr=LR, momentum=momentum, freeze=freeze,
                                 ae_model_path=ae_model_path)
    # move model to gpu
    model = model.to(device)
    print(model)
    # create trainer and evaluator
    if 'detection' in target_mode:
        trainer = create_detection_trainer(model, optimizer, device=device)
        evaluator = create_detection_evaluator(model, device=device)
    elif 'segmentation' in target_mode:
        trainer = create_segmentation_trainer(model, optimizer, device=device)
        evaluator = create_segmentation_evaluator(model, device=device)
    elif 'reconstruction' == target_mode.lower():
        trainer = create_autoencoder_trainer(model, optimizer, device=device)
        evaluator = create_autoencoder_evaluator(model, device=device)
    else:
        trainer, criterion = create_classification_trainer(model, optimizer, target_mode, device=device)
        evaluator = create_classification_evaluator(model, criterion, target_mode, device=device)

    # configuration of the engines
    trainer, evaluator = configure_engines( target_mode, model, optimizer, trainer, evaluator, train_dataloader,
                                           test_dataloader, config, display_info, fold,
                                           model_name, best_or_last, patience, model_path, device)
    # ---------------------------
    # Run training
    # ---------------------------
    if not only_eval:
        trainer.run(train_dataloader, max_epochs=EPOCHS)
    else:
        random.seed(0)

        # logging.info("===> Plotting one grad CAM")
        # ax, fig = plot_one_grad_cam(model, dataloader=test_dataloader, target_mode=target_mode, index=10)
        # plt.show()
        #
        # return
        target_mode = target_mode.lower()
        if 'single' in target_mode or 'multiple' in target_mode or 'fuzzy' in target_mode or 'classification' in target_mode:
            return evaluation_actions_classification(C, config, evaluator, external_data, fold, model, model_name,
                                                     model_path, target_mode, test_dataloader, train_dataloader,
                                                     device, best_or_last)
        elif target_mode == 'reconstruction':
            return evaluation_actions_reconstruction(C, config, evaluator, external_data, fold, model, model_name,
                                                     model_path, target_mode, test_dataloader, train_dataloader,
                                                     device, best_or_last)
        elif target_mode == 'detection':
            return evaluation_actions_detection(C, config, trainer, evaluator, external_data, fold, model, model_name,
                                                     model_path, target_mode, test_dataloader, train_dataloader,
                                                     device, best_or_last)

        else:
            raise Exception(f"Target mode = {target_mode} doen't have an evalution action.")

def evaluation_actions_reconstruction(C, config, evaluator, external_data, fold, model, model_name, model_path,
                                      target_mode, test_dataloader, train_dataloader, device, best_or_last):

    logging.info('Running evaluations Train and test (in that order).')
    evaluator.run(train_dataloader)
    logging.info(f"TRAIN: evaluator.state.metrics {evaluator.state.metrics}")
    evaluator.run(test_dataloader)
    logging.info(f"TEST: evaluator.state.metrics' {evaluator.state.metrics} ")
    # plotting the lattent space with (T-SNE)
    num_classes = C.labels.num_classes
    save_fig = True

    # loading model

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        logging.info('model loaded: ', model_path)
    else:
        if best_or_last == 'best':
            best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)
            if best_model_path is not None:
                best_model_path = os.path.join('best_models', best_model_path)
                model.load_state_dict(torch.load(best_model_path))
                logging.info('best model loaded: %s ' % best_model_path)
        else:
            last_model_path = get_last_checkpoint('models', fold, model_name, target_mode)
            if last_model_path is not None:
                last_model_path = os.path.join('models', last_model_path)
                model.load_state_dict(torch.load(last_model_path)['weights'])
                logging.info('last model loaded: %s' % model_path)
    plot_latent_space(model, num_classes=num_classes, device=device, data_loader=test_dataloader, save=save_fig,
                      dim_red='tsne')
    return 0

def evaluation_actions_detection(C, config, trainer, evaluator, external_data, fold, model, model_name, model_path,
                                        target_mode, test_dataloader, train_dataloader, device, best_or_last):
    logging.info('Running evaluations Train and test (in that order).')
    # evaluator.state.coco_evaluator = trainer.state.get_coco_evaluator()
    # evaluator.run(train_dataloader)
    # logging.info(f"TRAIN: evaluator.state.metrics {evaluator.state.metrics}")
    # evaluator.state.coco_evaluator = evaluator.state.get_coco_evaluator()
    # evaluator.run(test_dataloader)
    # logging.info(f"TEST: evaluator.state.metrics' {evaluator.state.metrics} ")
    # plotting one detection result
    # test_dataset = test_dataloader.dataset
    test_dataset = test_dataloader.dataset
    # get one image each
    mask_plot = "mask" in model_name
    import random
    for u in range(len(test_dataset)):
        N = random.randint(0, len(test_dataset)-1)
        image_test, label_test = test_dataset[u]
        fname = test_dataset.image_fnames[u]
        # get the prediction
        model.eval()
        with torch.no_grad():
            pred_test = model(image_test.unsqueeze(0).to(device))
        # plot the image and the prediction
        plot_detection(image_test, pred_test, label_test, C.labels.target_str_to_num, save=True, show=False, mask=mask_plot, suffix=f'{u}', fname=fname)
        #print('=====> fname', fname)
        #plot_detection(image_test, [label_test], label_test, C.labels.target_str_to_num, save=True, show=False, mask=mask_plot, suffix=f'{u}', fname=fname)
    return 0


def evaluation_actions_classification(C, config, evaluator, external_data, fold, model, model_name, model_path,
                                      target_mode, test_dataloader, train_dataloader, device, best_or_last):
    logging.info('dataset statistics')
    all_dataloader = get_dataloaders(config, batch=16, mode='all')
    all_labels = []
    # collect all labels in a list
    if os.path.exists('./dataset_statistics.csv'):
        df_all = pd.read_csv('./dataset_statistics.csv')
    else:
        for x, y in tqdm.tqdm(all_dataloader, desc="Collecting labels", total=len(all_dataloader)):
            all_labels.extend(y.tolist())
        df_all = pd.DataFrame(all_labels, columns=['label'])
        target_num_to_str = {v: k for k, v in C.labels.target_str_to_num.items()}
        df_all['label_name'] = df_all['label'].apply(lambda x: target_num_to_str[x])
        # save the dataset statistics
        df_all.to_csv('./dataset_statistics.csv', index=False)

    a = df_all.groupby('label_name').count()
    print(a)
    print(df_all.describe())

    # plot features of a layer
    logging.info("===> Plotting one grad CAM")
    # ax, fig = plot_one_grad_cam(model, dataloader=test_dataloader, target_mode=target_mode, index=10)
    # plt.show()
    # plot_classification_features(model, dataloader=test_dataloader, target_mode=target_mode, index=10, device=device)
    # plot_classification_features(model, dataloader=all_dataloader, target_mode=target_mode, index=2, device=device,
    #                              target_layer=['layer4.2.conv3', 'layer1.0.conv1'], fname='Malignant_117_IMAGIC_1608030190809.JPG')
    # os.abort()

    # sns.set(style="darkgrid")
    ax = sns.countplot(x="label_name", data=df_all, palette=['#432371',"#FAAE7B"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_ylabel('Number of instances')
    plt.show()
    os.abort()
    logging.info('Running evaluations Train and test (in that order).')
    evaluator.run(train_dataloader)
    logging.info(f"TRAIN: evaluator.state.metrics {evaluator.state.metrics}")
    evaluator.run(test_dataloader)
    logging.info(f"TEST: evaluator.state.metrics' {evaluator.state.metrics} ")
    _fold = fold if not external_data else 'external'
    predictions_fname = f'./predictions_fold={_fold}_{model_name}_{target_mode}.csv'
    if False:  # os.path.exists(predictions_fname):
        df = pd.read_csv(predictions_fname)
    else:
        # best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)
        # best_model_path = os.path.join('best_models', best_model_path)
        # model.load_state_dict(torch.load(best_model_path))
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
            logging.info('model loaded: ', model_path)
        else:
            best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)
            if best_model_path is not None:
                best_model_path = os.path.join('best_models', best_model_path)
                model.load_state_dict(torch.load(best_model_path))
                logging.info('best model loaded: ', model_path)
        df = predict_samples(model, test_dataloader, fold, target_mode, device=device)
        df.to_csv(predictions_fname, index=False)
    logging.info(df.head())
    logging.info('prediction_results.csv saved')
    df = error_analysis(df)
    # logging.info(f"prediction summary: {df['error'].describe()}")
    accTotal = float(len(df) - df['error'].sum()) / len(df)
    logging.info(f' Prediction acc: {accTotal}')
    class_names = list(C.labels.target_str_to_num.keys())
    report = classification_report(df['y_true'], df['y_pred'], labels=range(len(class_names)),
                                   target_names=class_names)
    logging.info(report)
    matrix = confusion_matrix(df['y_true'], df['y_pred'])
    accuracies = matrix.diagonal() / matrix.sum(axis=1)
    logging.info(' Accuracy per class:')
    for acc, class_name in zip(accuracies, class_names):
        logging.info(f'{class_name}: {acc}')
    logging.info('confusion matrix')
    disp = ConfusionMatrixDisplay(confusion_matrix=evaluator.state.metrics['cm'].numpy(),
                                  display_labels=class_names)
    disp.plot(xticks_rotation='vertical')
    plt.tight_layout()
    plt.show()
    return accTotal


if __name__ == "__main__":

    # main(target_mode='multiple', patience=None, epochs=100, fold=0)
    # main(target_mode='fuzzy', patience=15, epochs=100, fold=0)
    # main(target_mode='cropSingle', patience=15, epochs=100, fold=0, only_eval=True)
    # main(target_mode='classification', patience=15, epochs=100, fold=0, only_eval=True)

    # main(target_mode='cropSingle',  epochs=100, fold=0, batch_size=32, lr=0.001, model_name='resnet101', freeze='layer4.2.conv3', optimizer='ADAM', only_eval=True)
    #main(target_mode='multiple',  epochs=100, fold=0, batch_size=32, lr=0.00001, model_name='resnet101', freeze='layer4.2.conv3', optimizer='ADAM', only_eval=False)
    # main(target_mode='segmentation',  epochs=100000, fold=0, batch_size=16, lr=0.00001, model_name='unet', freeze='No', optimizer='ADAM', only_eval=False)
    main(target_mode='detection',  epochs=100, fold=0, batch_size=4, lr=0.000001, model_name='fasterresnet50', freeze='layer4.2.conv3', optimizer='ADAM', only_eval=False)
    # files = os.listdir(PATH)
    # accuracies = {f: 0 for f in files}
    # for f in files: #os.listdir(PATH):
    #     print(' ======================================== ')
    #     print(f)
    #     fold = int(f[f.index('fold=')+5])
    #     ff = os.path.join(PATH, f)
    #     try:
    #         acc = main(target_mode='cropSingle',  epochs=100, fold=fold, batch_size=32, lr=0.001,
    #              model_name='resnet101', freeze='layer4.2.conv3', optimizer='ADAM', only_eval=True, model_path=ff)
    #         accuracies[f] = acc
    #     except Exception as e:
    #         # logging.error(f'ERROR: {e}')
    #         print(f'cannot run fiel: {ff}')
    #         print(e)
    # data = {'accuracies': list(accuracies.values()), 'files': list(accuracies.keys())}
    # accuracies = pd.DataFrame(data=data)
    # accuracies.to_csv('accuracies.csv')

    # main(target_mode='cropSingle', patience=15, epochs=100, fold=0, config_file='config.ini', model_name='vgg19', only_eval=True)
    # main(target_mode='detectionSingle', model_name='faster_rcnn_resnet50_fpn', patience=15, epochs=100, fold=0)
    # main(target_mode='multiple', patience=15, epochs=100, fold=0)
    model_path = None
    # main(target_mode='multiple', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True)
    # main(target_mode='fuzzy', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True)
    # single training with split
    # main(target_mode='single', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True)
    # EXTERNAL DATA: single evaluation of external data
    # main(target_mode='single', patience=15, epochs=100, fold=0, model_path=model_path, only_eval=True, external_data=True)

    ## load model path
    # model = get_model("asd")
    # if model_path is not None:
    #     model.load_state_dict(torch.load(model_path)['model'])
    #     logging.info('loaded model', model_path)

    # evaluate models:
    # main(target_mode='cropSingle',  epochs=100, fold=0, batch_size=32, lr=0.001,
    #      model_name='resnet101', freeze='layer4.2.conv3', optimizer='ADAM', only_eval=True, model_path=model_path)

    # training classification autoencoder classifier
    # main(target_mode='reconstruction',  epochs=100, fold=0, batch_size=16, lr=1E-03, model_name='convcvae',
    #      freeze='layer4.0.conv3', optimizer='ADAM', only_eval=False)
    # main(target_mode='reconstruction',  epochs=200, fold=0, batch_size=16, lr=1E-06, model_name='convcvae',
    #      freeze='backbone', optimizer='ADAM', only_eval=True)
    # ae_model_path = -1# best_models/best_fold=0_convcvae_reconstruction_malignant_model_negval=-489.7608.pt'
    # main(target_mode='cropSingle',  epochs=100, fold=0, batch_size=32, lr=1E-6,
    #     model_name='aec', freeze='layer4.2.conv3', optimizer='ADAM', only_eval=False, model_path=model_path,
    #     ae_model_path=ae_model_path)
    # main(target_mode='cropSingle',  epochs=100, fold=0, batch_size=32, lr=1E-6,
    #     model_name='aec', freeze='encoder', optimizer='ADAM', only_eval=True, model_path=model_path,
    #     ae_model_path=ae_model_path)
    # 
    # training detection
    # main(target_mode='detection',  epochs=100, fold=0, batch_size=1, lr=0.000001, model_name='maskrcnn',
    #      freeze='layer4.2.conv3', optimizer='ADAM', only_eval=False)

    # main(target_mode='detection',  epochs=100, fold=0, batch_size=1, lr=0.000001, model_name='maskrcnn',
    #      freeze='layer4.2.conv3', optimizer='ADAM', only_eval=True, best_or_last='last')
    # training of autoencoders
    # main(target_mode='reconstruction',  epochs=200, fold=0, batch_size=16, lr=1E-06, model_name='convcvae',
    #      freeze='backbone', optimizer='ADAM', only_eval=False)

    # main(target_mode='reconstruction',  epochs=200, fold=0, batch_size=16, lr=1E-06, model_name='convcvae',
    #      freeze='backbone', optimizer='ADAM', only_eval=True)


    print('this is created from the browser :)')
