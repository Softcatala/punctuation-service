#/bin/sh
cd srv/web/
gunicorn  --workers=2 --graceful-timeout 90 --timeout 90 --threads=4 punctuation-service:app -b 0.0.0.0:8000
