mkdir -p ~/.streamlit/
echo "[theme]
primaryColor                = '#3C7290'
backgroundColor             = '#E5E5E5'
secondaryBackgroundColor    = '#fcf6f2'
textColor                   = '#3E738D'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml