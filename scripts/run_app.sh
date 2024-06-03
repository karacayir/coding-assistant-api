#!/bin/sh
#set -e

if ls | grep -q app.py
then
   echo "OK, you're in the correct folder";
else
   echo "NOT OK, please run the script at the same folder with the application";
   exit 1
fi

# in all cases
export FASTAPI_APP=app:app
export FASTAPI_PORT=$1
export PYTHONWARNINGS="ignore:Unverified HTTPS request"

COMMAND='uvicorn --host 0.0.0.0 --port $FASTAPI_PORT $FASTAPI_APP --log-config uvicorn_log_config.yml 2>&1'

echo "Starting FastAPI with $(eval echo $COMMAND)"
eval $COMMAND