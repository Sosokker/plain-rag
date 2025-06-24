### 🛠️ Project Tech & Environment Rules

* **Python Version:** `3.12`
* **Dependency Management:** [`uv`](https://github.com/astral-sh/uv) (fast, deterministic, PEP 582-compatible)
* **Backend Framework:** [`FastAPI`](https://fastapi.tiangolo.com/)
* **ORM:** [`SQLAlchemy 2.x`](https://docs.sqlalchemy.org/en/20/)
* **Migrations:** [`Alembic`](https://alembic.sqlalchemy.org/)
* **Authentication:** [`fastapi-users`](https://fastapi-users.github.io/fastapi-users/latest/) + [`fastapi-jwt-auth`](https://indominusbyte.github.io/fastapi-jwt-auth/)
* **Rate Limiting:** [`fastapi-limiter`](https://github.com/long2ice/fastapi-limiter)
* **Caching:** [`fastapi-cache`](https://github.com/long2ice/fastapi-cache)
* **Email Service:** [`fastapi-mail`](https://github.com/sabuhish/fastapi-mail)
* **Pagination:** [`fastapi-pagination`](https://github.com/uriyyo/fastapi-pagination)
* **LLM Layer:** [`litellm`](https://github.com/BerriAI/litellm)
* **Embedding Models:** [`Hugging Face Transformers`](https://huggingface.co/models)
* **Vector Store:** `pgvector` with PostgreSQL

Use pure dataclasses between in-app layers (as a transport objects). Use pydantic to validate user (external) input like web APIs. 

---

### 🧑‍💻 Python & Backend Code Quality Rules

#### 📦 Structure & Conventions

* **Follow modern `SQLAlchemy 2.0` best practices** (use `async engine`, `DeclarativeBase`, `SessionLocal()` pattern).
* **Separate concerns clearly:**

  * `models/`: SQLAlchemy models
  * `schemas/`: Pydantic models
  * `api/routes/`: FastAPI routers
  * `services/`: Business logic
  * `core/`: Settings, config, and utilities
  * `tests/`: Test suite

#### 🧹 Clean Code Principles

1. **Use Meaningful Names:** Functions, classes, variables, and routes should clearly communicate their intent.
2. **Avoid Overengineering:** YAGNI (You Aren’t Gonna Need It) — keep your code minimal, testable, and readable.
3. **Follow PEP 8 + Black Formatting:** Auto-format with `ruff`, lint with `ruff` or `flake8`.
4. **Use Type Hints Everywhere:** Both function arguments and return types must use type annotations.
5. **Use Docstrings:**

   * One-liner for simple functions.
   * Full docstring for public APIs and complex logic.
6. **Write Isolated, Testable Logic:** Favor pure functions where possible, especially in `services/`.
7. **Handle Exceptions Gracefully:**

   * Use `HTTPException` for expected FastAPI errors.
   * Log unexpected errors using `structlog`.
8. **Use Dependency Injection:** Use `Depends()` for shared logic (e.g., current user, DB session, rate limiter).

---

### 🧪 Testing Rules

* Use `pytest` as your testing framework.
* Coverage should include:

  * CRUD operations
  * API endpoints
  * Embedding & RAG pipeline logic
* Use `pytest-asyncio` for async route testing.
* Use fixtures for test data setup.

---

### 🔒 Security Practices

* Never store plaintext passwords — use hashing (`argon2`, `bcrypt` via `fastapi-users`).
* Sanitize file uploads & inputs — protect against injection.
* Use CORS middleware correctly (`allow_credentials`, `allow_methods`, etc.).
* Enable rate limiting on sensitive routes like login & upload.

---

### 🚀 Performance & Observability

* Add `structlog` structured logging to:

  * API entry/exit points
  * Query vector lookup latency
  * LLM response times
* Cache results where appropriate (`fastapi-cache`) — especially static vector responses.
* Stream LLM responses via FastAPI's `StreamingResponse`.