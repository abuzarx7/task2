#!/bin/bash
mkdir -p ~/.nltk_data
python -c "import nltk; nltk.download('punkt', download_dir='~/.nltk_data')"
