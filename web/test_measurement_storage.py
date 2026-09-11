import os
import tempfile
import time
import uuid

import app as target
from itsdangerous import BadSignature


def run():
    template_path = os.path.join(os.path.dirname(__file__),
                                 'templates', 'index.html')
    with open(template_path, encoding='utf-8') as template_file:
        template = template_file.read()
    assert 'id="btnSave"' not in template
    assert '开始测量即自动保存' in template
    assert 'saveMeasurement(data.save_token, saveGeneration)' in template

    old_upload = target.app.config['UPLOAD_FOLDER']
    old_secret = target.app.config['SAVE_TOKEN_SECRET']
    old_save = target._save_measurement_record
    old_lookup = target._lookup_measurement_code
    try:
        with tempfile.TemporaryDirectory() as upload_dir:
            target.app.config['UPLOAD_FOLDER'] = upload_dir
            target.app.config['SAVE_TOKEN_SECRET'] = 'test-secret-' * 4
            filename = f'result_multi_{uuid.uuid4().hex}.png'
            expired_name = f'result_{uuid.uuid4().hex}.png'
            expired_path = os.path.join(upload_dir, expired_name)
            with open(expired_path, 'wb') as image_file:
                image_file.write(b'expired')
            expired_time = time.time() - target.SAVE_TOKEN_TTL_SECONDS - 1
            os.utime(expired_path, (expired_time, expired_time))
            target._cleanup_expired_results()
            assert not os.path.exists(expired_path)

            result = {
                'foot_length': 255.2,
                'ball_width': 96.4,
                'heel_width': 67.1,
                'quality_grade': 'high',
                'dimension_confidence': {
                    'foot_length': 'high',
                    'ball_width': 'high',
                    'heel_width': 'medium',
                },
                'warnings': ['测试提示'],
                'result_image': f'/uploads/{filename}',
            }
            token = target._issue_save_token(result, 'left')
            decoded = target._read_save_token(token)
            assert decoded['foot_side'] == 'left'
            assert decoded['foot_length_mm'] == 255.2
            try:
                target._read_save_token(token + 'x')
                raise AssertionError('篡改后的凭证不应通过验证')
            except BadSignature:
                pass

            result_path = os.path.join(upload_dir, filename)
            with open(result_path, 'wb') as image_file:
                image_file.write(b'\x89PNG\r\n\x1a\nmock')
            target._save_measurement_record = (
                lambda data, image: 'FM-000001')
            target._lookup_measurement_code = (
                lambda nonce: 'FM-000001')

            client = target.app.test_client()
            missing_side = client.post('/api/measure', data={})
            assert missing_side.status_code == 400
            assert '左脚还是右脚' in missing_side.get_json()['error']
            response = client.post(
                '/api/measurements', json={'save_token': token})
            assert response.status_code == 200, response.get_json()
            assert response.get_json()['measurement_code'] == 'FM-000001'
            assert not os.path.exists(result_path)

            repeated = client.post(
                '/api/measurements', json={'save_token': token})
            assert repeated.status_code == 200, repeated.get_json()
            assert repeated.get_json()['already_saved'] is True
            assert repeated.get_json()['measurement_code'] == 'FM-000001'

            retry_name = f'result_{uuid.uuid4().hex}.png'
            retry_path = os.path.join(upload_dir, retry_name)
            with open(retry_path, 'wb') as image_file:
                image_file.write(b'keep-on-failure')
            retry_token = target._issue_save_token({
                **result,
                'result_image': f'/uploads/{retry_name}',
            }, 'right')

            def fail_save(data, image):
                raise RuntimeError('database unavailable')

            target._save_measurement_record = fail_save
            failed = client.post(
                '/api/measurements', json={'save_token': retry_token})
            assert failed.status_code == 503
            assert os.path.exists(retry_path)
    finally:
        target.app.config['UPLOAD_FOLDER'] = old_upload
        target.app.config['SAVE_TOKEN_SECRET'] = old_secret
        target._save_measurement_record = old_save
        target._lookup_measurement_code = old_lookup


if __name__ == '__main__':
    run()
    print('MEASUREMENT_STORAGE_OK')
