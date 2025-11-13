#/bin/sh
cd srv/web/
# if workers > 1 cache will not b shared across workers
rm -f $LOGDIR/id.txt
uvicorn punctuation-service:app --host 0.0.0.0 --port 8000 --workers 8 --timeout-keep-alive 90 --loop "uvloop"
