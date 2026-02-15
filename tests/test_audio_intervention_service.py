"""Tests for AudioInterventionService startup behavior."""

from src.core import audio_intervention


class _DummySocket:
    def close(self) -> None:
        pass


class _FakeBus:
    def __init__(self) -> None:
        self.publisher_calls: list[int] = []

    def create_subscriber(self, ports: list[int], topics: list[str]) -> _DummySocket:
        return _DummySocket()

    def create_publisher(self, port: int) -> _DummySocket:
        self.publisher_calls.append(port)
        return _DummySocket()

    def receive(self, socket: _DummySocket, timeout_ms: int = 500):
        return None

    def publish(self, socket: _DummySocket, topic: str, data: dict) -> None:
        pass


class _DummyIntervention:
    def __init__(self, *args, **kwargs) -> None:
        self._use_llm = False
        self._warning_gen = None
        self.last_tts_ms = None

    def should_intervene(self, analysis: dict) -> bool:
        return False

    def detect_scam_type(self, analysis: dict) -> str:
        return "generic_high_risk"

    def intervene(self, analysis: dict) -> None:
        return None


def test_start_does_not_create_conflicting_tactics_publisher(monkeypatch) -> None:
    """Intervention service should subscribe to tactics without rebinding port 5558."""
    monkeypatch.setattr(audio_intervention, "AudioIntervention", _DummyIntervention)

    bus = _FakeBus()
    service = audio_intervention.AudioInterventionService(bus=bus)

    # Avoid entering the infinite receive loop in this unit test.
    monkeypatch.setattr(service, "_main_loop", lambda: None)

    service.start()

    assert bus.publisher_calls == []
