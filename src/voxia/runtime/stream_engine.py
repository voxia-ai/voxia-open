class StreamingTTSEngine:

    def __init__(self, adapter):

        self.adapter = adapter

    def stream(self, request):

        features = self.adapter.prepare(request)

        for frame in self.adapter.stream(features):

            yield frame