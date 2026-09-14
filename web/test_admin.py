import os
from datetime import datetime, timezone
from unittest.mock import patch

import app as target


def main():
    original_rows = target._admin_measurement_rows
    original_export = target._admin_measurement_export
    original_image = target._admin_measurement_image
    row = {
        'measurement_code': 'FM-000029',
        'foot_side': 'left',
        'foot_length_mm': 273.7,
        'ball_width_mm': 98.3,
        'heel_width_mm': 58.6,
        'quality_grade': 'medium',
        'dimension_confidence': {'foot_length': 'medium'},
        'warnings': ['请核验轮廓'],
        'created_at': datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc),
        'expires_at': datetime(2028, 9, 14, 12, 0, tzinfo=timezone.utc),
    }
    export_row = tuple(row.values())
    try:
        target._admin_measurement_rows = lambda search, page: ([row], 1)
        target._admin_measurement_export = lambda search: [export_row]
        target._admin_measurement_image = lambda code: b'png-data'
        target._ADMIN_LOGIN_FAILURES.clear()
        with patch.dict(os.environ, {
                'ADMIN_USERNAME': 'maintainer',
                'ADMIN_PASSWORD': 'test-password',
                'ADMIN_DATABASE_URL': 'postgresql://reader/db'}, clear=False):
            client = target.app.test_client()
            assert client.get('/admin/measurements').status_code == 302
            assert client.post('/admin/login', data={
                'username': 'maintainer', 'password': 'wrong'
            }).status_code == 401
            login = client.post('/admin/login', data={
                'username': 'maintainer', 'password': 'test-password'
            })
            assert login.status_code == 302
            listing = client.get('/admin/measurements?q=FM-000029')
            assert listing.status_code == 200
            assert 'FM-000029'.encode() in listing.data
            image = client.get('/admin/measurements/FM-000029/outline.png')
            assert image.status_code == 200 and image.data == b'png-data'
            exported = client.get('/admin/measurements.csv')
            assert exported.status_code == 200
            assert 'FM-000029'.encode() in exported.data
    finally:
        target._admin_measurement_rows = original_rows
        target._admin_measurement_export = original_export
        target._admin_measurement_image = original_image
        target._ADMIN_LOGIN_FAILURES.clear()
    print('ADMIN_OK')


if __name__ == '__main__':
    main()
