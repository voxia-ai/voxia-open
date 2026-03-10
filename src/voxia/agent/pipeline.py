class VoicePipeline:

    def __init__(self, stt, llm, tts):

        self.stt = stt
        self.llm = llm
        self.tts = tts

    def run(self, audio):

        text = self.stt.transcribe(audio)

        reply = self.llm.generate(text)

        wav, sr = self.tts.speak(reply)

        return reply, wav, sr