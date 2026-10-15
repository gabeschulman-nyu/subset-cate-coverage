PY ?= python
export PYTHONPATH := src

.PHONY: all figures tables verify clean

all: figures tables

figures:
	$(PY) experiments/make_figure_benchmarks.py
	$(PY) experiments/make_figure_twopath.py
	$(PY) experiments/make_figure_collapse.py
	$(PY) experiments/cfm/make_figure_cfm_heads.py
	$(PY) experiments/cfm/make_figure_fixed_weight.py --head cnp
	$(PY) experiments/cfm/make_figure_fixed_weight.py --head latent

tables:
	cd docs/latex && $(PY) build_tex_tables.py
	$(PY) experiments/cfm/final_tables.py

# Confirms you have the artifacts we shipped, then rebuilds everything from
# them. The generators are seeded and the figures are byte-reproducible, so a
# clean run reproduces the paper's images and tables exactly.
verify:
	sha256sum -c artifacts.sha256
	$(MAKE) clean all

clean:
	rm -f docs/figures/*.pdf docs/figures/*.png docs/latex/tables/*.tex docs/cfm_arxiv_numbers.md
