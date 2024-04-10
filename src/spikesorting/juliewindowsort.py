import sys

import tkfilebrowser
from PyQt5.QtWidgets import QApplication
from windowsort.gui import get_default_directory, save_default_directory, MainWindow


def main():
    app = QApplication(sys.argv)

    # Load the default directory
    default_directory = get_default_directory()

    # Create a directory selection dialog
    # JULIE SPECIFIC - QFILEDIALOG DOESN'T WORK
    data_directory = tkfilebrowser.askopendirname(initialdir=default_directory, title="Select Data Directory")

    # Check if the user pressed cancel (i.e., data_directory is empty)
    if not data_directory:
        print("No directory selected. Exiting.")
        sys.exit()

    # Save the selected directory as the new default
    save_default_directory(data_directory)

    print("Loading App")
    mainWin = MainWindow(data_directory)
    mainWin.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
