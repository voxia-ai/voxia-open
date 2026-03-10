class StreamEngine:

    def __init__(self, adapter):

        self.adapter = adapter

    def generate(self, request):

        for frame in self.adapter.stream(request):

            yield frame