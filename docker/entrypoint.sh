#/bin/sh
cd srv/web/
# if workers > 1 cache will not b shared across workers
gunicorn  --workers=1 --graceful-timeout 90 --timeout 90 --threads=8 punctuation-service:app -b 0.0.0.0:8000
