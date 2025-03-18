#!/bin/bash
mkdir -p $HOME/nltk_data
python -c "import nltk; nltk.download('punkt', download_dir='$HOME/nltk_data'); nltk.download('stopwords', download_dir='$HOME/nltk_data')"
