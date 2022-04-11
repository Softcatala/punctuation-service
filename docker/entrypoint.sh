#/bin/sh
cd srv/web/
gunicorn style-service:app -b 0.0.0.0:8000 --workers=2
