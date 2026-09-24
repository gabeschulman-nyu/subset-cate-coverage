PY ?= python
export PYTHONPATH := src

.PHONY: all figures tables verify clean

all: figures tables

figures:
	$(PY) experiments/make_figure_benchmarks.py
	$(PY) experiments/make_figure_twopath.py
	$(PY) experiments/make_figure_collapse.py

# threearm.tex is emitted by the script that also draws its figure, so the
# table and the figure cannot disagree about the numbers.
tables:
	cd docs/latex && $(PY) build_tex_tables.py
	$(PY) -c "import sys; sys.path.insert(0,'experiments'); \
	  import make_figure_threearm as m; \
	  open('docs/latex/tables/threearm.tex','w').write(m.table())"

# Confirms you have the artifacts we shipped, then rebuilds everything from
# them. The generators are seeded and the figures are byte-reproducible, so a
# clean run reproduces the paper's images and tables exactly.
verify:
	sha256sum -c artifacts.sha256
	$(MAKE) clean all

clean:
	rm -f docs/figures/*.pdf docs/figures/*.png docs/latex/tables/*.tex
