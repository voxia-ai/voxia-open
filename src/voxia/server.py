from __future__ import annotations

import warnings


def run_server(
    *,
    model: str,
    device: str = "cpu",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    warnings.filterwarnings("ignore", category=FutureWarning)

    from voxia import TTS
    from voxia.api import create_app
    import uvicorn

    tts = TTS.load(model, format="sbv2", device=device, strict=False)
    app = create_app(tts=tts, model_path=model)

    uvicorn.run(app, host=host, port=int(port), log_level="info")