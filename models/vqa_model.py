import lightning.pytorch as pl


class LitVqaModel(pl.LightningModule):
    def __init__(self, ocr_encoder, img_encoder, text_decoder):
        super().__init__()
        self.ocr_encoder = ocr_encoder
        self.img_encoder = img_encoder
        self.text_decoder = text_decoder

    def training_step(self, batch, batch_idx):
        # TODO
        pass

    def configure_optimizers(self):
        # TODO
        pass
