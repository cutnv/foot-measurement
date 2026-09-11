import os
import tempfile
import uuid

import app as target


def run():
    if not os.environ.get('DATABASE_URL'):
        raise RuntimeError('请先设置测试数据库 DATABASE_URL')
    old_upload = target.app.config['UPLOAD_FOLDER']
    old_secret = target.app.config['SAVE_TOKEN_SECRET']
    try:
        with tempfile.TemporaryDirectory() as upload_dir:
            target.app.config['UPLOAD_FOLDER'] = upload_dir
            target.app.config['SAVE_TOKEN_SECRET'] = 'integration-test-' * 3
            filename = f'result_multi_{uuid.uuid4().hex}.png'
            with open(os.path.join(upload_dir, filename), 'wb') as image_file:
                image_file.write(b'\x89PNG\r\n\x1a\nmock')
            token = target._issue_save_token({
                'foot_length': 254.8,
                'ball_width': 95.3,
                'heel_width': 66.2,
                'quality_grade': 'high',
                'dimension_confidence': {
                    'foot_length': 'high',
                    'ball_width': 'high',
                    'heel_width': 'high',
                },
                'warnings': [],
                'result_image': f'/uploads/{filename}',
            }, 'right')
            client = target.app.test_client()
            first = client.post(
                '/api/measurements', json={'save_token': token})
            assert first.status_code == 200, first.get_json()
            code = first.get_json()['measurement_code']
            assert code.startswith('FM-')
            repeated = client.post(
                '/api/measurements', json={'save_token': token})
            assert repeated.status_code == 200, repeated.get_json()
            assert repeated.get_json()['measurement_code'] == code
            assert repeated.get_json()['already_saved'] is True
    finally:
        target.app.config['UPLOAD_FOLDER'] = old_upload
        target.app.config['SAVE_TOKEN_SECRET'] = old_secret


if __name__ == '__main__':
    run()
    print('DATABASE_INTEGRATION_OK')
