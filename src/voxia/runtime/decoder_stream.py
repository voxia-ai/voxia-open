import torch


class DecoderStreamer:

    def __init__(self, model):

        self.model = model

    @torch.inference_mode()
    def stream_decode(self, latent):

        step = 16

        length = latent.shape[-1]

        for i in range(0, length, step):

            frame = latent[..., i:i+step]

            audio = self.model.decoder(frame)

            yield audio