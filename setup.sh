#!/bin/bash

mkdir -p ~/.streamlit/

echo "\
[general]
email = \"ameenkhan2016yo@gmail.com\"

[server]
headless = true
enableCORS = false
port = \$PORT
" > ~/.streamlit/config.toml
