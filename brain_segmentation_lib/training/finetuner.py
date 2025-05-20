from .trainer import Trainer

class FineTuner(Trainer):
    def fine_tune(self, epochs, freeze_until=10):
        for layer in self.model.layers[:freeze_until]:
            layer.trainable = False
        self.model.fit(self.dataloader, epochs=epochs)