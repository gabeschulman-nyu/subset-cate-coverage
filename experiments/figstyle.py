"""Shared matplotlib style for the paper's figures.

Figures are drawn at their printed size and included without scaling, so font
sizes print as set. Nimbus Roman (a Times clone) matches the ICLR template's
body text; mathtext uses STIX. Colours are Okabe-Ito, each paired with a line
style and marker for greyscale printing.
"""

from __future__ import annotations

import datetime as _datetime

import matplotlib as mpl

# ICLR text width, inches.
WIDTH = 5.5

# Okabe-Ito, ordered for separation in the first four entries.
OKABE_ITO = ["#0072B2",  # blue
             "#D55E00",  # vermillion
             "#009E73",  # bluish green
             "#CC79A7",  # reddish purple
             "#E69F00",  # orange
             "#56B4E9",  # sky blue
             "#000000",  # black
             "#F0E442"]  # yellow

# Line styles and markers paired with the palette.
STYLES = [(0, ()), (0, (4, 1.6)), (0, (1, 1.2)), (0, (5, 1.4, 1, 1.4)),
          (0, (3, 1.2, 1, 1.2, 1, 1.2)), (0, (7, 2)), (0, (2, 1)), (0, (6, 1.5, 2, 1.5))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

GREY = "#666666"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # 9pt text, 8pt ticks and legends.
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.2,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "grid.color": "#B0B0B0",
        "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        # No tight bbox, which would trim the canvas and make LaTeX rescale
        # the figure. Constrained layout fits the content inside the canvas.
        "savefig.bbox": None,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.dpi": 400,
        "figure.dpi": 150,
        "pdf.fonttype": 42,          # embed TrueType, no Type-3 subsetting
        "ps.fonttype": 42,
        "pdf.compression": 6,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=OKABE_ITO)


def save(fig, path, *, report=True):
    """Save PDF and PNG without identifying metadata; print size and smallest font."""
    # A fixed CreationDate makes the PDF byte-reproducible.
    meta = {"Creator": "", "Producer": "", "Author": "", "Title": "",
            "Subject": "", "Keywords": "",
            "CreationDate": _datetime.datetime(2000, 1, 1)}
    fig.savefig(path, metadata=meta)
    fig.savefig(path.replace(".pdf", ".png"), metadata={"Software": ""})
    if report:
        w, h = fig.get_size_inches()
        # Smallest rendered font over tick labels, axis labels and legends,
        # including figure-level legends.
        sizes = [t.get_fontsize() for ax in fig.axes
                 for t in (ax.get_xticklabels() + ax.get_yticklabels()
                           + [ax.xaxis.label, ax.yaxis.label])
                 if t.get_text()]
        lgs = [ax.get_legend() for ax in fig.axes if ax.get_legend()] + list(fig.legends)
        lg = [t.get_fontsize() for g in lgs for t in g.get_texts()]
        smallest = min(sizes + lg) if (sizes or lg) else float("nan")
        print(f"  {path.split('/')[-1]:26s} {w:.2f} x {h:.2f} in   "
              f"min font {smallest:.1f}pt")
