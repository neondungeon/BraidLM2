"""
main.py — BraidLM entry point
"""
import sys
from PySide6.QtWidgets import QApplication
from window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("BraidLM")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
