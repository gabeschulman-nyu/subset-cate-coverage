"""One matplotlib style for every figure in the paper.

Figures are built at their FINAL printed size and included with
\\includegraphics[width=\\textwidth] and no further scaling. Rendering large and
letting LaTeX shrink is what makes tick labels illegible: a 10pt label in a
10-inch figure scaled to 5.5 inches prints at 5.5pt. Build at 5.5 inches and
9pt stays 9pt.

Font is Nimbus Roman, the URW Times clone, which is metrically compatible with
the ``times`` package the ICLR template loads, so figure text matches body
text. Mathtext uses STIX for the same reason.

The palette is Okabe--Ito, which is colourblind-safe by construction. Every
series additionally carries a distinct line style and marker, so the figures
survive greyscale printing without relying on hue.
"""

from __future__ import annotations

import matplotlib as mpl

# ICLR single column. Figures are built at this width; nothing is scaled twice.
WIDTH = 5.5

# Okabe--Ito, ordered for maximum separation in the first four entries.
OKABE_ITO = ["#0072B2",  # blue
             "#D55E00",  # vermillion
             "#009E73",  # bluish green
             "#CC79A7",  # reddish purple
             "#E69F00",  # orange
             "#56B4E9",  # sky blue
             "#000000",  # black
             "#F0E442"]  # yellow

# Paired with the palette so hue is never the only cue.
STYLES = [(0, ()), (0, (4, 1.6)), (0, (1, 1.2)), (0, (5, 1.4, 1, 1.4)),
          (0, (3, 1.2, 1, 1.2, 1, 1.2)), (0, (7, 2)), (0, (2, 1)), (0, (6, 1.5, 2, 1.5))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

GREY = "#666666"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # 9pt body, 8pt ticks: the floor once the figure is not rescaled.
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
        # NOT bbox="tight": a tight bbox trims the canvas, so the saved PDF is
        # narrower than 5.5in and \includegraphics[width=\textwidth] scales it
        # back up, enlarging every font by the same factor. Constrained layout
        # fits the content INSIDE the 5.5in canvas instead, so the saved width
        # is exactly the printed width and 9pt prints as 9pt.
        "savefig.bbox": None,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.dpi": 400,
        "figure.dpi": 150,
        "pdf.fonttype": 42,          # embed TrueType, no Type-3 subsetting
        "ps.fonttype": 42,
        # Double-blind: matplotlib otherwise writes the username into
        # /Creator and the local path into some backends' metadata.
        "pdf.compression": 6,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=OKABE_ITO)


def save(fig, path, *, report=True):
    """Save PDF and PNG, strip identifying metadata, and report the size that
    will actually be printed."""
    meta = {"Creator": "", "Producer": "", "Author": "", "Title": "",
            "Subject": "", "Keywords": ""}
    fig.savefig(path, metadata=meta)
    fig.savefig(path.replace(".pdf", ".png"), metadata={"Software": ""})
    if report:
        w, h = fig.get_size_inches()
        # Smallest font actually rendered, so the 8pt floor can be checked
        # rather than assumed.
        sizes = [t.get_fontsize() for ax in fig.axes
                 for t in (ax.get_xticklabels() + ax.get_yticklabels()
                           + [ax.xaxis.label, ax.yaxis.label])
                 if t.get_text()]
        lg = [t.get_fontsize() for ax in fig.axes
              if ax.get_legend() for t in ax.get_legend().get_texts()]
        smallest = min(sizes + lg) if (sizes or lg) else float("nan")
        print(f"  {path.split('/')[-1]:26s} {w:.2f} x {h:.2f} in   "
              f"min font {smallest:.1f}pt")
