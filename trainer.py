import torch


class Trainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.epoch = 0

    def train_on_batch(self, src_batch, trg_batch, opt):
        batch = self._merge_batches(src_batch, trg_batch)
        metadata = {'epoch': self.epoch, 'n_epochs': self.n_epochs}
        loss = self.loss(self.model, batch, **metadata)

        opt.zero_grad()
        loss.backward()
        opt.step()

    def _merge_batches(self, src_batch, trg_batch):
        src_images, src_classes = src_batch
        trg_images, trg_classes = trg_batch
        batch = dict()
        # batch['src_images'] = src_images
        # batch['trg_images'] = trg_images
        # batch['src_classes'] = src_classes
        # batch['trg_classes'] = trg_classes

        batch['images'] = torch.cat([src_images, trg_images], dim=0)
        batch['true_classes'] = torch.cat([src_classes, trg_classes], dim=0)
        batch['domains'] = torch.cat([torch.zeros(len(src_classes)), torch.ones(len(src_classes))], dim=0)
        return batch

    def fit(self, src_data, trg_data, n_epochs=1000, steps_per_epoch=100, val_freq=1,
            opt='adam', opt_kwargs=None, validation_data=None, metrics=None):
        self.n_epochs = n_epochs

        if opt_kwargs is None:
            opt_kwargs = dict()

        if opt == 'adam':
            opt = torch.optim.Adam(self.model.parameters(), **opt_kwargs)
        else:
            raise NotImplementedError

        for self.epoch in range(self.epoch, n_epochs):
            for step, (src_batch, trg_batch) in enumerate(zip(src_data, trg_data)):
                if step == steps_per_epoch:
                    break
                self.train_on_batch(src_batch, trg_batch, opt)

            if metrics is not None and self.epoch % val_freq == 0:
                src_val_data, trg_val_data = validation_data
                if src_val_data is not None:
                    src_metrics = self.score(src_val_data, metrics)
                if trg_val_data is not None:
                    trg_metrics = self.score(trg_val_data, metrics)

    def score(self, data, metrics):
        images, true_classes = data
        pred_classes = self.predict(images)
        return {metric.name: metric(pred_classes, true_classes) for metric in metrics}

    def predict_on_batch(self, batch):
        pass

    def predict(self, data):
        predictions = []
        for batch in data:
            predictions.append(self.predict_on_batch(batch))
        return torch.stack(predictions)
