class Trainer:
    def __init__(self, model, images, masks, loss, metrics):
        self.model = model
        self.images = images
        self.masks = masks
        self.model.compile(optimizer='adam', loss=loss, metrics=metrics)

    def train(self, epochs):
        self.model.fit(self.images, self.masks, epochs=epochs, batch_size=16)
