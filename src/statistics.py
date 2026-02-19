from matplotlib.patches import Patch

legend_patches = [
    Patch(color=color, label=name)
    for name, color in [
        ("Water", "#419bdf"),
        ("Trees", "#397d49"),
        ("Crops", "#e49635"),
        ("Built Area", "#c4281b"),
        ("Bare Ground", "#a59b8f"),
        ("Rangeland", "#e3e2c3"),
    ]
]

color_dict = {
    '1': '#419bdf',   # Water
    '2': '#397d49',   # Trees
    '5': '#e49635',   # Crops
    '7': '#c4281b',   # Built area
    '8': '#a59b8f',   # Bare ground
    '11': '#e3e2c3',  # Rangeland
}
