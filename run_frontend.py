import os
import sys

import streamlit.web.cli as stcli


def main():
    sys.argv = ["streamlit", "run", "frontend/app.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 