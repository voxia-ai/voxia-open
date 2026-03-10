class VoiceAgent:

    def __init__(self, pipeline):

        self.pipeline = pipeline

    def chat(self, audio):

        text, wav, sr = self.pipeline.run(audio)

        return {
            "text": text,
            "audio": wav,
            "sample_rate": sr
        }