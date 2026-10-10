# Contributing to Chatterbox

Thank you for your interest in contributing to the Chatterbox Autonomous Speech Engine! We welcome issues, bug reports, feature requests, and pull requests from developers and researchers worldwide.

---

## 1. Code of Conduct

Contributors are expected to adhere to professional, respectful, and collaborative behavior in all project interactions. Harassment, abuse, or discriminatory language of any kind will not be tolerated.

---

## 2. Development Setup

### Fork and Clone
```bash
git clone https://github.com/crrrowz/chatterbox.git
cd chatterbox
```

### Environment Configuration
We recommend using a dedicated virtual environment with Python 3.10+:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
pip install pytest httpx
```

---

## 3. Architecture Guidelines

Chatterbox is organized with clear layer boundaries:
- `chatterbox.engine`: Core neural inference wrappers, voice registries, and thread locks. Must remain stateless across multiple requests.
- `chatterbox.tasks`: Autonomous queue execution and explicit state machine (`RECEIVED -> VALIDATING -> GENERATING -> VERIFYING -> COMPLETED`).
- `chatterbox.artifacts`: File storage, audio converters, and forensic verification logic.
- `chatterbox.api`: FastAPI route handlers, Pydantic request/response schemas, and endpoints.
- `chatterbox.web`: Static assets and studio dashboard for human operators.

When submitting changes:
1. **Preserve Model Safety**: Never remove or bypass the `threading.Lock` surrounding `_model.generate(...)`.
2. **Never Return Corrupted Data**: A job must never be marked `COMPLETED` unless the generated artifact passes `AudioVerifier.verify_file(...)`.
3. **Keep Client Adapters Decoupled**: The core engine must remain orchestrator-agnostic.

---

## 4. Running Tests

Before opening a pull request, verify that the entire test suite passes:
```bash
pytest tests/ -v
```

If you modify or add new endpoints or chunking logic, include corresponding unit and integration tests under `tests/`.

---

## 5. Submitting a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Push to your branch and open a Pull Request against `main`.
4. Describe the problem your PR solves, how you tested it, and any architectural considerations.
