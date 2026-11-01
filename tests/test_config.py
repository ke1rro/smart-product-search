from pathlib import Path

import pytest
from decouple import AutoConfig, Config, RepositoryEmpty

from smart_product_search.config import PROJECT_ROOT, load_settings


@pytest.fixture(autouse=True)
def isolate_configuration_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ordinary configuration tests independent of local credentials and dotenv files."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setattr("smart_product_search.config.config", Config(RepositoryEmpty()))


def test_load_project_configuration_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load the checked-in configuration without requiring an API credential."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    settings = load_settings()

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.index_path == PROJECT_ROOT / "data/index/products.sqlite3"
    assert (settings.retrieval_k, settings.rerank_k, settings.classify_k) == (100, 50, 10)
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.llm_model == ""
    assert settings.openrouter_api_key == ""


def test_resolve_defaults_independently_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep project paths stable when called from a different working directory."""
    configuration = tmp_path / "settings.toml"
    configuration.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = load_settings(configuration)

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.index_path == PROJECT_ROOT / "data/index/products.sqlite3"
    assert not (tmp_path / "data").exists()


def test_load_custom_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply explicit TOML values and preserve absolute storage paths."""
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    configuration = tmp_path / "settings.toml"
    configuration.write_text(
        f'data_dir = "{tmp_path.as_posix()}"\n'
        'index_path = "data/index/custom.sqlite3"\n'
        "retrieval_k = 20\nrerank_k = 12\nclassify_k = 3\n"
        'embedding_model = "example/embedding"\nllm_model = "example/model:free"\n',
        encoding="utf-8",
    )

    settings = load_settings(configuration)

    assert settings.data_dir == tmp_path
    assert settings.index_path == PROJECT_ROOT / "data/index/custom.sqlite3"
    assert (settings.retrieval_k, settings.rerank_k, settings.classify_k) == (20, 12, 3)
    assert settings.embedding_model == "example/embedding"
    assert settings.llm_model == "example/model:free"


def test_read_environment_overrides_without_exposing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use environment credentials and model overrides without exposing credentials in repr."""
    configuration = tmp_path / "settings.toml"
    configuration.write_text('llm_model = "example/toml:free"\n', encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "example/environment:free")

    settings = load_settings(configuration)

    assert settings.openrouter_api_key == "test-only-not-a-real-key"
    assert settings.llm_model == "example/environment:free"
    assert "test-only-not-a-real-key" not in repr(settings)


def test_read_dotenv_without_credentials_in_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read the project dotenv file with environment variables taking precedence."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setattr("smart_product_search.config.config", AutoConfig(search_path=str(tmp_path)))
    configuration = tmp_path / "settings.toml"
    configuration.write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=dotenv-test-only\nOPENROUTER_MODEL=example/dotenv:free\n",
        encoding="utf-8",
    )

    settings = load_settings(configuration)

    assert settings.openrouter_api_key == "dotenv-test-only"
    assert settings.llm_model == "example/dotenv:free"
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-test-only")
    assert load_settings(configuration).openrouter_api_key == "environment-test-only"


@pytest.mark.parametrize(
    "content",
    [
        "retrieval_k = 0",
        "rerank_k = -1",
        "classify_k = 0",
        "retrieval_k = 5",
        "classify_k = 51",
        "retrieval_k = true",
        'rerank_k = "50"',
        "classify_k = 1.5",
        'embedding_model = ""',
        "llm_model = 123",
        'data_dir = ""',
        "index_path = false",
        'openrouter_api_key = "must-not-be-stored-here"',
        "retrieve_k = 100",
    ],
)
def test_reject_invalid_configuration(tmp_path: Path, content: str) -> None:
    """Reject invalid settings at the external configuration boundary."""
    configuration = tmp_path / "settings.toml"
    configuration.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        load_settings(configuration)
