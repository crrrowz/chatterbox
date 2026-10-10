import pytest
from chatterbox.engine.service import SpeechEngine, ValidationError
from chatterbox.engine.config import EngineConfig


@pytest.fixture
def engine():
    cfg = EngineConfig(max_text_length=100)
    return SpeechEngine(config=cfg)


def test_validation_empty_text(engine):
    with pytest.raises(ValidationError, match="must not be empty"):
        engine.validate_request(text="", language="ar")

    with pytest.raises(ValidationError, match="must not be empty"):
        engine.validate_request(text="   ", language="ar")


def test_validation_text_length_exceeded(engine):
    with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
        engine.validate_request(text="A" * 101, language="ar")


def test_validation_unsupported_language(engine):
    with pytest.raises(ValidationError, match="Unsupported language code 'xx'"):
        engine.validate_request(text="Hello", language="xx")


def test_validation_supported_languages(engine):
    # Should not raise for valid languages
    engine.validate_request(text="مرحبا بكم", language="ar")
    engine.validate_request(text="Hello world", language="en")
    engine.validate_request(text="Bonjour", language="fr")


def test_validation_numeric_bounds(engine):
    with pytest.raises(ValidationError, match="Exaggeration"):
        engine.validate_request(text="Test", language="en", exaggeration=2.5)

    with pytest.raises(ValidationError, match="Temperature"):
        engine.validate_request(text="Test", language="en", temperature=0.01)

    with pytest.raises(ValidationError, match="CFG weight"):
        engine.validate_request(text="Test", language="en", cfg_weight=-0.5)

    with pytest.raises(ValidationError, match="Repetition penalty"):
        engine.validate_request(text="Test", language="en", repetition_penalty=3.0)
