"""
renderer.py
"""

import os
from IPython.display import clear_output


def render(
    altitude,
    velocity,
    wind,
    action,
    reward,
    step,
    max_altitude=1000.0,
    is_jupyter=False
):


    # Clear previous frame

    if is_jupyter:
        clear_output(wait=True)
    else:
        os.system("cls" if os.name == "nt" else "clear")

    terminal_lines = 40

    # Automatically zoom near landing

    if altitude > 150.0:
        display_max = max_altitude
        zoom_text = "[ CAMERA : WIDE ANGLE (1000 m) ]"
    else:
        display_max = 150.0
        zoom_text = "[ CAMERA : TARGET APPROACH (150 m) ]"

    probe_position = int(
        (altitude / display_max) * terminal_lines
    )

    probe_position = max(
        0,
        min(terminal_lines, probe_position)
    )

    wind_text = [
        "~ Calm ~",
        "Gusty",
        "Adrian Gale"
    ]

    thrust_text = (
        "[####] ON"
        if action == 1
        else "[    ] OFF"
    )

    print(
        f"T+{step:04d} | "
        f"ALT: {altitude:7.1f} m | "
        f"VEL: {velocity:7.2f} m/s | "
        f"REWARD: {reward:8.2f}"
    )

    print(
        f"THRUST: {thrust_text} | "
        f"WIND: {wind_text[wind]}"
    )

    print(zoom_text)

    print("-" * 75)

    # Draw altitude scale and probe

    for i in range(terminal_lines, -1, -1):

        if i == probe_position:

            if action == 1:

                print("           /\\")
                print("           ||")
                print("          /WW\\")
                print("           ||   <-- Spin Drive")

            else:

                print("           /\\")
                print("           ||")
                print("          /--\\")
                print("")

        else:

            if i % 10 == 0:

                altitude_mark = int(
                    (i / terminal_lines) * display_max
                )

                print(
                    f"{altitude_mark:4d} m +---------------------"
                )

            else:

                print("       |")

    print("=" * 75)
    print("                 TAUMOEBA LANDING TARGET")
    print("=" * 75)