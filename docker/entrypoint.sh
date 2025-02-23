#/bin/sh
cd srv/web/
# if workers > 1 cache will not b shared across workers
uvicorn punctuation-service:app --host 0.0.0.0 --port 8000 --workers 2 --timeout-keep-alive 90 --loop "uvloop"
