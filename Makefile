PY ?= python
export PYTHONPATH := src

.PHONY: all figures tables verify clean

all: figures tables

figures:
	$(PY) experiments/make_figure_benchmarks.py
	$(PY) experiments/make_figure_twopath.py
	$(PY) experiments/make_figure_collapse.py

tables:
	cd docs/latex && $(PY) build_tex_tables.py

# Confirms you have the artifacts we shipped, then rebuilds everything from
# them. The generators are seeded and the figures are byte-reproducible, so a
# clean run reproduces the paper's images and tables exactly.
verify:
	sha256sum -c artifacts.sha256
	$(MAKE) clean all

clean:
	rm -f docs/figures/*.pdf docs/figures/*.png docs/latex/tables/*.tex
