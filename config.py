import os
from dataclasses import dataclass


### define all app-wide configuration here,
### should not be accessed and changed directly hence leading "__"
@dataclass
class __AppConfig:
    """app-wide configurations"""

    title = "Computer Vision - Landmark Classifier"
    theme = "freddyaboulton/dracula_revamped"
    hf_repo_id = "sssingh/landmark-classifier-pt"
    hf_weights_file = "model_transfer.pt"
    css = "style.css"
    model = None
    classes = [
        "Haleakala_National_Park",
        "Mount_Rainier_National_Park",
        "Ljubljana_Castle",
        "Dead_Sea",
        "Wroclaws_Dwarves",
        "London_Olympic_Stadium",
        "Niagara_Falls",
        "Stonehenge",
        "Grand_Canyon",
        "Golden_Gate_Bridge",
        "Edinburgh_Castle",
        "Mount_Rushmore_National_Memorial",
        "Kantanagar_Temple",
        "Yellowstone_National_Park",
        "Terminal_Tower",
        "Central_Park",
        "Eiffel_Tower",
        "Changdeokgung",
        "Delicate_Arch",
        "Vienna_City_Hall",
        "Matterhorn",
        "Taj_Mahal",
        "Moscow_Raceway",
        "Externsteine",
        "Soreq_Cave",
        "Banff_National_Park",
        "Pont_du_Gard",
        "Seattle_Japanese_Garden",
        "Sydney_Harbour_Bridge",
        "Petronas_Towers",
        "Brooklyn_Bridge",
        "Washington_Monument",
        "Hanging_Temple",
        "Sydney_Opera_House",
        "Great_Barrier_Reef",
        "Monumento_a_la_Revolucion",
        "Badlands_National_Park",
        "Atomium",
        "Forth_Bridge",
        "Gateway_of_India",
        "Stockholm_City_Hall",
        "Machu_Picchu",
        "Death_Valley_National_Park",
        "Gullfoss_Falls",
        "Trevi_Fountain",
        "Temple_of_Heaven",
        "Great_Wall_of_China",
        "Prague_Astronomical_Clock",
        "Whitby_Abbey",
        "Temple_of_Olympian_Zeus",
    ]
    mean = [0.4624, 0.4711, 0.4668]
    std = [0.2592, 0.2600, 0.2925]


app_config = __AppConfig()
