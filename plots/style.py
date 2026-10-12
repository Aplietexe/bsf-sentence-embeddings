import os

import matplotlib.pyplot as plt

PAPER, INK, MUTED, LINE = "#FFFCF8", "#26272B", "#67666B", "#DED7D0"
RED, TEAL, GOLD, SLATE, GRAY, PURPLE = "#A43651", "#237B79", "#AD7226", "#6E7286", "#99908B", "#86628E"
DATASETS = {"20ng": "20 Newsgroups", "arxiv": "arXiv", "dbpedia": "DBpedia"}
BOLD = {"family": "Avenir Next", "weight": "bold"}
SHORT_NAMES = {
    "comp.graphics": "Graphics",
    "comp.os.ms-windows.misc": "Windows",
    "comp.sys.ibm.pc.hardware": "PC hardware",
    "comp.sys.mac.hardware": "Mac hardware",
    "comp.windows.x": "X11",
    "rec.autos": "Cars",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine",
    "sci.space": "Space",
    "talk.politics.guns": "Guns",
    "talk.politics.mideast": "Middle East",
    "talk.politics.misc": "Politics",
    "alt.atheism": "Atheism",
    "soc.religion.christian": "Christianity",
    "talk.religion.misc": "Religion",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Avenir", "DejaVu Sans"],
        "font.serif": ["Georgia", "DejaVu Serif"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": LINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "grid.color": LINE,
        "grid.linewidth": 0.65,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}.png", dpi=220)
    plt.close(fig)
