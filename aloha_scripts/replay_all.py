#!/usr/bin/env python3

import sys
import tty
import termios
from argparse import ArgumentParser
from pathlib import Path
from replay_episodes import replay
from rich.table import Table
from rich.console import Console
from rich.progress import Progress


def format_table(passes, fails):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Episode", style="cyan", no_wrap=True)
    table.add_column("Pass", style="green")
    table.add_column("Fail", style="red")

    for key in passes:
        pas, fail = passes[key], fails[key]
        if pas > 0:
            table.add_row(key, str(pas), str(fail))

    for key in passes:
        pas, fail = passes[key], fails[key]
        if fail > 0:
            table.add_row('[red]' + key, str(pas), str(fail))

    console = Console()
    console.print(table)


# Function to read a single character from input
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return char


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_dir', help='dataset directory')
    args = parser.parse_args()

    files = list(Path(args.dataset_dir).glob("*.hdf5"))
    passes = {str(file): 0 for file in files}
    fails = {str(file): 0 for file in files}
    i = 0
    file = files[i]
    total_to_process = len(files)
    processed = 0

    print("Press <space> to start, q to quit")
    char = getch()

    if char == "q":
        exit()

    for file in files:

        print(f"Replaying {file}")
        replay(str(file))

        # Prompt the user to press a key
        print("Press <space> for pass, q to quit, or any key for fail")
        char = getch()
        processed += 1

        # Check if the character is a space
        if char == " ":
            passes[str(file)] = 1
        # elif char == "\n" or char == "\r":
        elif char == "q":
            break
        else:
            fails[str(file)] += 1

        # output status
        print(f"{processed} / {total_to_process}")
        format_table(passes, fails)