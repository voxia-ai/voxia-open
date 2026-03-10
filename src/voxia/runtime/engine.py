class RuntimeEngine:

    def __init__(self, pipeline):

        self.pipeline = pipeline

    def run(self, request):

        return self.pipeline.run(request)

    def stream(self, request):

        for chunk in self.pipeline.stream(request):

            yield chunk