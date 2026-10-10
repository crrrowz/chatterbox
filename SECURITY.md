# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## Reporting a Vulnerability

We take the security of the Chatterbox Autonomous Speech Engine seriously. If you discover a security vulnerability, please follow responsible disclosure guidelines.

### Reporting Procedure
- **Do not open a public issue.**
- Email the maintainers directly or use GitHub's private vulnerability reporting feature.
- Include detailed steps to reproduce the issue, along with proof-of-concept code if applicable.
- Allow the maintainers reasonable time to investigate, remediate, and publish a fix before disclosing the issue publicly.

---

## Security Invariants & Principles

1. **Local-First Default**: By default, the server binds to `127.0.0.1:8000`. Do not expose the service publicly without explicit authentication enabled (`CHATTERBOX_ENABLE_AUTH=true`).
2. **Path Traversal Protection**: Job IDs and artifact filenames are strictly sanitized to prevent directory traversal attacks (`job_id` characters are restricted to alphanumeric and underscores).
3. **Information Disclosure Prevention**: Internal Python exception stack traces and server-side model paths are never returned in public API error responses. Clients receive structured diagnostic codes without exposing host environment internals.
4. **Denial-of-Service Boundaries**: Input text length is bounded (`CHATTERBOX_MAX_TEXT_LEN`), queue sizes are constrained (`CHATTERBOX_MAX_QUEUE_SIZE`), and timeouts are enforced.
