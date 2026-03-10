from dataclasses import dataclass


@dataclass
class VoiceResponse:

    text: str
    audio: bytes