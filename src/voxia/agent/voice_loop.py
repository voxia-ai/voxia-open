import sounddevice as sd


class VoiceLoop:

    def __init__(self, agent):

        self.agent = agent

    def start(self):

        while True:

            audio = self.record()

            response = self.agent.chat(audio)

            self.play(response["audio"], response["sample_rate"])

    def record(self):

        print("Listening...")

        audio = sd.rec(16000 * 3, samplerate=16000, channels=1)

        sd.wait()

        return audio

    def play(self, wav, sr):

        sd.play(wav, sr)

        sd.wait()