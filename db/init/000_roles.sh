#!/bin/sh
set -eu

: "${APP_DB_PASSWORD:?APP_DB_PASSWORD is required}"
: "${READER_DB_PASSWORD:?READER_DB_PASSWORD is required}"

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=ON_ERROR_STOP=1 \
  --set=app_password="$APP_DB_PASSWORD" \
  --set=reader_password="$READER_DB_PASSWORD" <<'SQL'
CREATE ROLE foot_app LOGIN PASSWORD :'app_password';
CREATE ROLE foot_reader LOGIN PASSWORD :'reader_password';
SQL
