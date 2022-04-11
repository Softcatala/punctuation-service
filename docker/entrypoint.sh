#/bin/sh
cd srv/web/
gunicorn punctuation-service:app -b 0.0.0.0:8000 --workers=2
