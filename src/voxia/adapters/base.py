class ModelAdapter:

    def prepare(self, request):
        raise NotImplementedError

    def infer(self, features):
        raise NotImplementedError

    def stream(self, request):
        raise NotImplementedError
    
class SBV2Adapter(ModelAdapter):

    def __init__(self, model):

        self.model = model

    def prepare(self, request):

        return request.text

    def infer(self, features):

        return self.model.infer_text(features)

    def stream(self, request):

        wav = self.infer(self.prepare(request))

        yield wav