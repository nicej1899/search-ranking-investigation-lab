PY ?= .venv/bin/python
all: prepare train cases rankers select investigate bench report test
prepare:     ; $(PY) -m srlab.data
train:       ; $(PY) -m srlab.train --epochs 30
cases:       ; $(PY) -m srlab.queries
rankers:     ; $(PY) -m srlab.ranker
select:      ; $(PY) -m srlab.cases
investigate: ; $(PY) -m srlab.agent
bench:       ; $(PY) -m srlab.bench
report:      ; $(PY) -m srlab.report
test:        ; $(PY) -m pytest -q
web:         ; cd web && npm ci && npm run build
serve:       ; $(PY) -m srlab.server
.PHONY: all prepare train cases rankers select investigate bench report test web serve
