.PHONY: lint test integration-test docs verify-s0 eval eval-live

lint:
	uv run ruff check src tests scripts evaluation
	uv run mypy src

test:
	uv run pytest tests/unit -v

integration-test:
	uv run pytest tests/integration -v

docs:
	uv run python scripts/gen_protocol_doc.py

# 压缩质量评测：离线门禁（零成本，可进 CI）；--live 需 API Key
eval:
	uv run python scripts/run_compact_eval.py
	uv run python scripts/run_e2e_eval.py

eval-live:
	uv run python scripts/run_compact_eval.py --live
	uv run python scripts/run_e2e_eval.py --live

verify-s0:
	uv sync --frozen
	uv run ruff check src tests scripts evaluation
	uv run mypy src
	uv run pytest tests/unit -v
	uv run pytest tests/integration -k ping -v
	uv run python scripts/gen_protocol_doc.py --check
