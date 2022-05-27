import torch
from skinbot.dataset import get_dataloaders 
from skinbot.config import read_config
from skinbot.models import get_model

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

def main():
    log_interval = 1
    log_interval = 1
    config = read_config()
    root_dir = config["DATASET"]["root"]
    # prepare dataset
    test_dataloader = get_dataloaders(config, batch=16, mode='test')
    train_dataloader = get_dataloaders(config, batch=16, mode='train')
    

    # prepare models
    model, optimizer = get_model('resnet101', optimizer='SGD')
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = create_supervised_trainer(model, optimizer, criterion)
    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
            print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(test_dataloader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    trainer.run(train_dataloader, max_epochs=5)



if __name__ == "__main__":
    main()
    

