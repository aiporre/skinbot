import copy
import skinbot.logging as logging
import math
import os

import pandas as pd
import torch
import torchvision.models.detection
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, global_step_from_engine
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, RunningAverage, DiceCoefficient, IoU, mIoU
from ignite.engine import EventEnum
from torch import distributed as dist
import numpy as np

from skinbot.losses import MulticlassLoss, CosineLoss, EuclideanLoss
# from skinbot.transformers import num_classes, target_weights
from skinbot.utils import validate_target_mode, get_log_path, make_patches, join_patches, load_models
from skinbot.config import Config

from skinbot.torchvisionrefs.coco_eval import CocoEvaluator
from skinbot.torchvisionrefs.coco_utils import convert_to_coco_api

C = Config()


class CheckpointEvents(EventEnum):
    """
    Custom events defined by user
    """
    SAVE_BEST = 'save_best'
    SAVE_LAST = 'save_last'


def get_last_checkpoint(path_models, fold, model_name, target_mode, by_iteration=False):
    prefix = f"last_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}_checkpoint"
    if by_iteration:
        iterations = [p.split('_')[-1].split('.')[0] for p in os.listdir(path_models) if
                      p.endswith('.pt') and p.startswith(prefix)]
        iterations = [int(ii) for ii in iterations if ii.isnumeric()]
        if len(iterations) == 0:
            return None
        last_iteration = max(iterations)
        return f"{prefix}_{last_iteration}.pt"
    else:
        checkpoints = [p for p in os.listdir(path_models) if p.endswith('.pt') and p.startswith(prefix)]
        # sort checkpoints by creation time
        checkpoints = sorted(checkpoints, key=lambda x: os.path.getctime(os.path.join(path_models, x)))
        # return the last checkpoint created
        if len(checkpoints) == 0:
            return None
        return checkpoints[-1]


def get_best_iteration(path_models, fold, model_name, target_mode):
    prefix = f"best_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}_model"
    iterations = [p.split('=')[-1].split('.pt')[0] for p in os.listdir(path_models) if
                  p.endswith('.pt') and p.startswith(prefix)]
    if len(iterations) == 0:
        return None
    iterations = [float(ii) for ii in iterations]
    last_iteration = max(iterations)
    last_iteration_end = f"={last_iteration:.4f}.pt"
    best_model_path = [p for p in os.listdir(path_models) if p.endswith(last_iteration_end) and p.startswith(prefix)][0]
    return best_model_path


def keep_best_two(path_models, fold, model_name, target_mode):
    prefix = f"best_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}_model"
    iterations = [p.split('=')[-1].split('.pt')[0] for p in os.listdir(path_models) if
                  p.endswith('.pt') and p.startswith(prefix)]
    if len(iterations) < 2:
        return
    iterations = [float(ii) for ii in iterations]
    iterations = sorted(iterations, reverse=True)
    iterations_to_remove = iterations[2:]
    for iteration in iterations_to_remove:
        iteration_end = f"={iteration:.4f}.pt"
        best_model_path = [p for p in os.listdir(path_models) if p.endswith(iteration_end) and p.startswith(prefix)][0]
        os.remove(os.path.join(path_models, best_model_path))


def prepare_batch(batch, device=None):
    x, y = batch
    x = list(xx.to(device, non_blocking=True) for xx in x)
    y = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in y]
    return x, y

def prepare_batch_seg(batch, device=None):
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x,y



def reduce_dict(loss_dict):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        names = []
        values = []
        for k, v in loss_dict.items():
            names.append(k)
            values.append(v)
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_losses = {k: v for k, v in zip(names, values)}
    return reduced_losses


def get_loss_keys(model, dataloader):
    batch = next(iter(dataloader))
    x, y = prepare_batch(batch)
    model.train()
    loss_dict = model(x, y)
    loss_dict_reduced = reduce_dict(loss_dict)
    return list(loss_dict_reduced.keys())


def create_detection_trainer(model, optimizer, device=None):
    def update_model(engine, batch):
        model.train()

        x, y = copy.deepcopy(batch)
        x_process, y_process = prepare_batch(batch, device=device)

        loss_dict = model(x_process, y_process)
        loss_sum = sum(value for value in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # update the model weights
        engine.state.optimizer.zero_grad()
        # if the loss is infinite, the use have to know about it
        if not math.isfinite(loss_value):
            logging.info("Loss is {}, resetting loss and skipping training iteration".format(loss_value))
            logging.info(f'Loss values were: %f {loss_dict_reduced}.')
            logging.info(f"Input labels were:  {[yy['labels'] for yy in y]}")
            logging.info(f"Input boxes were: ', {[yy['boxes'] for yy in y]}")
            loss_dict_reduced = {k: torch.tensor(0) for k, v in loss_dict_reduced.items()}
            loss_value = 0
        else:
            loss_sum.backward()
            engine.state.optimizer.step()

        if hasattr(engine.state, 'warmup_scheduler') and engine.state.warmup_scheduler is not None:
            engine.state.warmup_scheduler.step()

        return x, y, loss_dict_reduced

    engine = Engine(update_model)
    engine.state.optimizer = optimizer
    return engine


def create_detection_evaluator(model, device=None):
    def update_model(engine, batch):
        model.eval()
        x, y = prepare_batch(batch, device=device)
        x_process = copy.deepcopy(x)

        if torch.has_cuda:
            torch.cuda.synchronize()
        with torch.no_grad():
            y_pred = model(x_process)

        y_pred = [{k: v.to(device) for k, v in t.items()} for t in y_pred]

        res = {yy["image_id"].item(): yy_pred for yy, yy_pred in zip(y, y_pred)}
        engine.state.coco_evaluator.update(res)

        x_process = y_pred = None

        return x, y, res

    return Engine(update_model)


def create_segmentation_trainer(model, optimizer, device=None):
    # make loss according to target mode
    # compute inverse of class weights
    v_max = max(C.labels.target_weights.values())
    target_values_norm = [v_max / v for v in C.labels.target_weights.values()]
    target_weights_tensor = torch.tensor(target_values_norm, dtype=torch.float32, device=device)
    # define criterion 2D cross-entropy
    criterion = torch.nn.CrossEntropyLoss()
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    return trainer


def create_segmentation_evaluator(model, device=None):
    def dice_pre(output):
        y_pred, y = output  # (B, Cls, W, H) , (B, W, H)
        y = torch.flatten(y)  # (B*W*H)

        y_pred = torch.softmax(y_pred, dim=1)  # (B, Cls, W, H)
        y_pred = torch.argmax(y_pred, dim=1)  # (B, 1, W, H)
        y_pred = torch.flatten(y_pred)  # (B*W*H)
        y_pred = torch.nn.functional.one_hot(y_pred, num_classes=C.labels.num_classes).float()  # (B*W*H, Cls)
        # y_pred must be one-hot
        # y integers within [0,C)
        return y_pred, y

    cm = ConfusionMatrix(num_classes=C.labels.num_classes, output_transform=dice_pre)

    val_metrics = {
        "Dice": DiceCoefficient(cm),
        'IoU': IoU(cm),
        "mIoU": mIoU(cm)
    }

    def evaluate_step(engine, batch):
        model.eval()
        if torch.has_cuda:
            torch.cuda.synchronize()

        batch = prepare_batch_seg(batch, device)

        with torch.no_grad():
            x, y = batch
            # remove B=1
            x = x[0]
            patch_size = C.segmentation.patch_size
            overlap = C.segmentation.overlap
            patches, patch_size = make_patches(x, patch_size, overlap=overlap, device=device)
            pred_patches = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    x_patch = torch.unsqueeze(patches[i, j, :, :], dim=0)
                    pred_patch = model(x_patch)
                    pred_patches.append(torch.squeeze(pred_patch, dim=0))
            pred_patches = torch.stack(pred_patches, dim=0)
            # channels = prediction channels is the num of classes
            channels = C.labels.num_classes
            pred_patches = torch.reshape(pred_patches,
                                      (patches.shape[0], patches.shape[1], channels, patch_size[0],patch_size[1]))
            H, W = x.shape[-2], x.shape[-1]
            y_pred = join_patches(pred_patches, (channels, H, W), patch_size, overlap, device=device)
            # B =1 again
            y_pred = torch.unsqueeze(y_pred, dim=0)
            return y_pred, y

    # evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

    evaluator = Engine(evaluate_step)
    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_classification_trainer(model, optimizer, target_mode, device=None):
    # make loss according to target mode
    if 'single' in target_mode.lower():
        # compute inverse of class weights
        v_max = max(C.labels.target_weights.values())
        target_values_norm = [v_max / v for v in C.labels.target_weights.values()]
        target_weights_tensor = torch.tensor(target_values_norm, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=target_weights_tensor)
    elif 'multiple' in target_mode.lower():
        criterion = MulticlassLoss()
    elif 'fuzzy' in target_mode.lower():
        criterion = CosineLoss()
    else:
        raise ValueError(f"target_mode={target_mode} is not supported")

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    return trainer, criterion


def create_classification_evaluator(model, criterion, target_mode, device=None):
    def pred_in_prob(output):
        ''' convert prediction and target to one-hot vector '''
        y_pred, y = output
        y_pred_prob = torch.softmax(y_pred, dim=1)
        y_pred_class = torch.argmax(y_pred, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=C.labels.num_classes)
        # TODO: fix evaluation broken
        if validate_target_mode(target_mode, ['fuzzy', 'multiple']):
            y_argmax = torch.argmax(y, dim=1)
        elif 'single' in target_mode.lower():
            y_argmax = y.long()
        else:
            raise ValueError(f"target_mode={target_mode} is not supported")
        y_onehot = torch.nn.functional.one_hot(y_argmax, num_classes=C.labels.num_classes)
        return y_pred_onehot, y_onehot

    def pred_in_onehot(output):
        ''' convert prediction to one-hot vector  and taget into single label '''
        y_pred, y = output
        y_pred_class = torch.argmax(y_pred, dim=1)
        y_pred_onehot = torch.nn.functional.one_hot(y_pred_class, num_classes=C.labels.num_classes)

        if 'fuzzy' in target_mode.lower() or 'multiple' in target_mode.lower():
            #  target_mode in ['fuzzy', 'multiple']:
            y_argmax = torch.argmax(y, dim=1)
        elif 'single' in target_mode.lower():
            y_argmax = y.long()
        else:
            raise ValueError(f"target_mode={target_mode} is not supported")
        return y_pred_onehot, y_argmax

    def pred_thresholded(output):
        y_pred, y = output
        y_pred_prob = torch.sigmoid(y_pred)  # torch.softmax(y_pred, dim=1)
        y_pred = (y_pred_prob > 0.5).float()
        y = (y > 0.0).float()
        return y_pred, y

    val_metrics = {
        "accuracy": Accuracy() if not validate_target_mode(target_mode, ['fuzzy', 'multiple']) else Accuracy(
            output_transform=pred_thresholded),
        "nll": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=C.labels.num_classes, output_transform=pred_in_onehot),
        'cosine': Loss(CosineLoss()) if validate_target_mode(target_mode, ['fuzzy', 'multiple']) else Loss(
            torch.nn.CrossEntropyLoss()),
        'euclidean': Loss(EuclideanLoss()) if validate_target_mode(target_mode, ['fuzzy', 'multiple']) else Loss(
            torch.nn.CrossEntropyLoss()),
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    return evaluator


def configure_engines_classification(target_mode,
                                     model,
                                     optimizer,
                                     trainer,
                                     evaluator,
                                     train_dataloader,
                                     test_dataloader,
                                     config,
                                     display_info,
                                     fold,
                                     model_name,
                                     best_or_last,
                                     patience,
                                     model_path,
                                     device):
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    if display_info and torch.cuda.is_available():
        from ignite.contrib.metrics import GpuInfo

        GpuInfo().attach(trainer, name='gpu')
    if config['LOGGER']['logtofile'] == 'True':
        log_path = get_log_path(config)
        log_file_handler = open(log_path, 'w')
        pbar = ProgressBar(persist=True, file=log_file_handler)
    else:
        pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    # @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    # def log_training_loss(trainer):
    # print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_training_results(engine):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        # print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg accuracy: {avg_accuracy:.2f} "
            f"Avg loss: {avg_nll:.2f} "
            f"Avg Cosine: {metrics['cosine']:.2f}"
            f"Avg Euclidean: {metrics['euclidean']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=2))
    def log_validation_results(engine):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        # print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} "
            f"Avg accuracy: {avg_accuracy:.2f} "
            f"Avg loss: {avg_nll:.2f} "
            f"Avg Cosine {metrics['cosine']:.2f}"
            f"Avg Euclidean: {metrics['euclidean']:.2f}")

        pbar.n = pbar.last_print_n = 0
        evaluator.fire_event(CheckpointEvents.SAVE_BEST)

    to_save = {"weights": model, "optimizer": optimizer}
    handler_ckpt = Checkpoint(
        to_save,
        save_handler=DiskSaver('models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f"last_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}",
    )
    if best_or_last == 'last':
        last_checkpoint_path = get_last_checkpoint('models', fold, model_name, target_mode)
        if last_checkpoint_path is not None:
            last_checkpoint_path = os.path.join('models', last_checkpoint_path)
            to_load = to_save
            handler_ckpt.load_objects(to_load=to_load, checkpoint=last_checkpoint_path)
            logging.info(f'loaded last checkpoint {last_checkpoint_path}')

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), handler_ckpt)

    ## early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    if patience is not None:
        early_stop_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)

    to_save = {'model': model}
    handler_best = Checkpoint(
        to_save,
        save_handler=DiskSaver('best_models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f"best_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}",
        score_name="accuracy",
        # global_step_transform=global_step_from_engine(trainer)
    )

    if best_or_last == 'best':
        # get the best model path
        if model_path is not None:
            best_model_path = model_path
        else:
            # keeps the best_models directory clean
            keep_best_two('best_models', fold, model_name, target_mode)
            # get the best model path
            best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)

        # if success then load the best model
        if best_model_path is not None:
            best_model_path = os.path.join('best_models', best_model_path)
            to_load = to_save
            # model.load_state_dict(torch.load(best_model_path))
            load_models(model, best_model_path, device)
            to_save['model'] = model
            logging.info(f'Loaded best model {best_model_path}')
            handler_best.load_objects(to_load=to_load, checkpoint=best_model_path)
        else:
            logging.info(f'No best model found. starting from scratch')
    evaluator.register_events(*CheckpointEvents)
    evaluator.add_event_handler(CheckpointEvents.SAVE_BEST, handler_best)

    return trainer, evaluator


def configure_engines_detection(target_mode,
                                model,
                                optimizer,
                                trainer,
                                evaluator,
                                train_dataloader,
                                test_dataloader,
                                config,
                                display_info,
                                fold,
                                model_name,
                                best_or_last,
                                patience,
                                model_path,
                                device):
    from itertools import chain

    # configure evaluator coco api wrapper functions
    test_dataset = list(chain.from_iterable(zip(*batch) for batch in iter(test_dataloader)))
    coco_api_test_dataset = convert_to_coco_api(test_dataset)
    train_dataset = list(chain.from_iterable(zip(*batch) for batch in iter(train_dataloader)))
    coco_api_train_dataset = convert_to_coco_api(train_dataset)

    @evaluator.on(Events.COMPLETED)
    def on_evaluation_completed(engine):
        # gather the stats from all processes
        engine.state.coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        engine.state.coco_evaluator.accumulate()
        engine.state.coco_evaluator.summarize()
        # loading metric values (using hard-code values from coco eval api)
        # Average precision and average recall sets in this order:
        # 1. Iou 0.5 to 0.95 (all areas)
        # 2. Iou 0.5 (all areas)
        # 3. Iou 0.76 (all areas)
        # 4. Iou 0.5 to 0.95 (medium areas)
        # 5. Iou 0.5 to 0.95 (large areas)
        engine.state.metrics = {}
        for k, coco_eval in engine.state.coco_evaluator.coco_eval.items():
            if len(coco_eval.stats) > 0:
                engine.state.metrics[f'IoU_precision_0.5_{k}'] = coco_eval.stats[1]

    def infer_ioutypes_coco_api(_model):
        if isinstance(_model, torchvision.models.detection.FasterRCNN):
            ioutypes = ['bbox']
        elif isinstance(_model, torchvision.models.detection.MaskRCNN):
            ioutypes = ['bbox', 'segm']
        return ioutypes

    # configure logging progress bar
    loss_keys = get_loss_keys(model, train_dataloader)

    class RATrans:
        def __init__(self, k):
            self.k = k

        def __call__(self, output):
            x, y, loss = output
            return loss[self.k].item()

    for k in loss_keys:
        RunningAverage(output_transform=RATrans(k)).attach(trainer, k)
    RunningAverage(output_transform=lambda output: sum(loss for loss in output[2].values()).item()).attach(trainer,
                                                                                                           "loss")

    if display_info and torch.cuda.is_available():
        from ignite.contrib.metrics import GpuInfo
        GpuInfo().attach(trainer, name='gpu')

    if config['LOGGER']['logtofile'] == 'True':
        log_path = get_log_path(config)
        log_file_handler = open(log_path, 'w')
        pbar = ProgressBar(persist=True, file=log_file_handler)
    else:
        pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_training_results(engine):
        evaluator.state.coco_evaluator = CocoEvaluator(coco_api_train_dataset, infer_ioutypes_coco_api(model))
        evaluator.run(train_dataloader)

        metrics = evaluator.state.metrics
        # # print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        IoU_precision_0p5_bbox = metrics["IoU_precision_0.5_bbox"]
        # avg_nll = metrics["nll"]
        pbar.log_message(
            f"Training Results - Epoch: {engine.state.epoch} "
            #     f"Avg accuracy: {avg_accuracy:.2f} "
            #     f"Avg loss: {avg_nll:.2f} "
            #     f"Avg Cosine: {metrics['cosine']:.2f}"
            f"Avg Prec IoU@0.5 bbox: {IoU_precision_0p5_bbox:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_validation_results(engine):
        evaluator.state.coco_evaluator = CocoEvaluator(coco_api_test_dataset, infer_ioutypes_coco_api(model))
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        # # print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        IoU_precision_0p5_bbox = metrics["IoU_precision_0.5_bbox"]
        # avg_nll = metrics["nll"]
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} "
            #     f"Avg accuracy: {avg_accuracy:.2f} "
            #     f"Avg loss: {avg_nll:.2f} "
            #     f"Avg Cosine: {metrics['cosine']:.2f}"
            f"Avg Prec IoU@0.5 bbox: {IoU_precision_0p5_bbox:.2f}")
        evaluator.fire_event(CheckpointEvents.SAVE_BEST)

    to_save = {"weights": model, "optimizer": optimizer}
    handler_ckpt = Checkpoint(
        to_save,
        save_handler=DiskSaver('models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f"last_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}",
    )
    if best_or_last == 'last':
        last_checkpoint_path = get_last_checkpoint('models', fold, model_name, target_mode)
        if last_checkpoint_path is not None:
            last_checkpoint_path = os.path.join('models', last_checkpoint_path)
            to_load = to_save
            handler_ckpt.load_objects(to_load=to_load, checkpoint=last_checkpoint_path)
            logging.info(f'loaded last checkpoint {last_checkpoint_path}')

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), handler_ckpt)

    ## early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    if patience is not None:
        early_stop_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)

    to_save = {'model': model}
    handler_best = Checkpoint(
        to_save,
        save_handler=DiskSaver('best_models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f"best_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}",
        score_name="IoU_precision_0.5_bbox",
        # global_step_transform=global_step_from_engine(trainer)
    )

    if best_or_last == 'best':
        # get the best model path
        if model_path is not None:
            best_model_path = model_path
        else:
            # keeps the best_models directory clean
            keep_best_two('best_models', fold, model_name, target_mode)
            # get the best model path
            best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)

        # if success then load the best model
        if best_model_path is not None:
            best_model_path = os.path.join('best_models', best_model_path)
            to_load = to_save
            # model.load_state_dict(torch.load(best_model_path))
            load_models(model, best_model_path, device)
            to_save['model'] = model
            logging.info(f'Loaded best model {best_model_path}')
            handler_best.load_objects(to_load=to_load, checkpoint=best_model_path)
        else:
            logging.info(f'No best model found. starting from scratch')

    evaluator.register_events(*CheckpointEvents)
    evaluator.add_event_handler(CheckpointEvents.SAVE_BEST, handler_best)

    return trainer, evaluator


def configure_engines_segmentation(target_mode,
                                   model,
                                   optimizer,
                                   trainer,
                                   evaluator,
                                   train_dataloader,
                                   test_dataloader,
                                   config,
                                   display_info,
                                   fold,
                                   model_name,
                                   best_or_last,
                                   patience,
                                   model_path,
                                   device):
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    if display_info and torch.cuda.is_available():
        from ignite.contrib.metrics import GpuInfo

        GpuInfo().attach(trainer, name='gpu')
    if config['LOGGER']['logtofile'] == 'True':
        log_path = get_log_path(config)
        log_file_handler = open(log_path, 'w')
        pbar = ProgressBar(persist=True, file=log_file_handler)
    else:
        pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    # @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    # def log_training_loss(trainer):
    # print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_training_results(engine):
        print('running trainer evaluation.....')
        from torch.utils.data import DataLoader
        train_dataloader_validation = DataLoader(dataset=train_dataloader.dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=train_dataloader.num_workers)
        evaluator.run(train_dataloader_validation)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  ")
        avg_dice = metrics["Dice"].mean().item()
        avg_iou = metrics["IoU"].mean().item()
        avg_miou = metrics["mIoU"]
        evaluator.state.metrics["TrainDice"] = avg_dice
        logging.info(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg Dice: {avg_dice:.2f} "
            f"Avg IoU: {avg_iou:.2f} "
            f"Avg mIoU: {avg_miou:.2f}")
        print(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg Dice: {avg_dice:.2f} "
            f"Avg IoU: {avg_iou:.2f} "
            f"Avg mIoU: {avg_miou:.2f}")
        logging.info(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg Dice: {avg_dice:.2f} "
            f"Avg IoU: {avg_iou:.2f} "
            f"Avg mIoU: {avg_miou:.2f}")
        stats = {"Dice": [v.item() for v in metrics['Dice']],
                 "IoU": [v.item() for v in metrics['IoU']]}
        labels = [c for c in C.labels.target_str_to_num.keys()]
        stats = pd.DataFrame(stats, index=labels)
        logging.info(f"Stats per class: \n{stats}")
        print(f"Stats per class: \n{stats}")

    @trainer.on(Events.EPOCH_COMPLETED(every=2))
    def log_validation_results(engine):
        print('running validation ....')
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        # print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")
        avg_dice = metrics["Dice"].mean().item()
        avg_iou = metrics["IoU"].mean().item()
        avg_miou = metrics["mIoU"]
        evaluator.state.metrics["ValDice"] = avg_dice
        print(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg Dice: {avg_dice:.2f} "
            f"Avg IoU: {avg_iou:.2f} "
            f"Avg mIoU: {avg_miou:.2f}")
        logging.info(
            f"Training Results - Epoch: {engine.state.epoch} "
            f"Avg Dice: {avg_dice:.2f} "
            f"Avg IoU: {avg_iou:.2f} "
            f"Avg mIoU: {avg_miou:.2f}")
        stats = {"Dice": [v.item() for v in metrics['Dice']],
                 "IoU": [v.item() for v in metrics['IoU']]}
        labels = [c for c in C.labels.target_str_to_num.keys()]
        stats = pd.DataFrame(stats, index=labels)
        logging.info(f"Stats per class: \n{stats}")
        print(f"Stats per class: \n{stats}")
        pbar.n = pbar.last_print_n = 0
        evaluator.fire_event(CheckpointEvents.SAVE_BEST)

    to_save = {"weights": model, "optimizer": optimizer}
    handler_ckpt = Checkpoint(
        to_save,
        save_handler=DiskSaver('models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f"last_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}",
    )
    if best_or_last == 'last':
        last_checkpoint_path = get_last_checkpoint('models', fold, model_name, target_mode)
        if last_checkpoint_path is not None:
            last_checkpoint_path = os.path.join('models', last_checkpoint_path)
            to_load = to_save
            handler_ckpt.load_objects(to_load=to_load, checkpoint=last_checkpoint_path)
            logging.info(f'loaded last checkpoint {last_checkpoint_path}')

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=100), handler_ckpt)

    ## early stopping
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    if patience is not None:
        early_stop_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler)

    to_save = {'model': model}
    handler_best = Checkpoint(
        to_save,
        save_handler=DiskSaver('best_models', create_dir=True, require_empty=False),
        n_saved=2,
        filename_prefix=f"best_fold={fold}_{model_name}_{target_mode}_{C.label_setting()}",
        score_name="ValDice",
        # global_step_transform=global_step_from_engine(trainer)
    )

    if best_or_last == 'best':
        # get the best model path
        if model_path is not None:
            best_model_path = model_path
        else:
            # keeps the best_models directory clean
            keep_best_two('best_models', fold, model_name, target_mode)
            # get the best model path
            best_model_path = get_best_iteration('best_models', fold, model_name, target_mode)

        # if success then load the best model
        if best_model_path is not None:
            best_model_path = os.path.join('best_models', best_model_path)
            to_load = to_save
            # model.load_state_dict(torch.load(best_model_path))
            model = load_models(model, best_model_path, device)
            to_save['model'] = model
            logging.info(f'Loaded best model {best_model_path}')
            try:
                handler_best.load_objects(to_load=to_load, checkpoint=best_model_path)
            except RuntimeError as e:
                if torch.cuda.is_available():
                    state_dict = torch.load(best_model_path, map_location="cuda:0")
                else:
                    state_dict = torch.load(best_model_path, map_location='cpu')

                handler_best.load_objects(to_load=to_load, checkpoint={'model': state_dict})
        else:
            logging.info(f'No best model found. starting from scratch')
    evaluator.register_events(*CheckpointEvents)
    evaluator.add_event_handler(CheckpointEvents.SAVE_BEST, handler_best)

    return trainer, evaluator


def configure_engines(target_mode, *args, **kwargs):
    if 'detection' in target_mode.lower():
        return configure_engines_detection(target_mode, *args, **kwargs)
    elif 'segmentation' in target_mode.lower():
        return configure_engines_segmentation(target_mode, *args, **kwargs)
    else:
        return configure_engines_classification(target_mode, *args, **kwargs)
