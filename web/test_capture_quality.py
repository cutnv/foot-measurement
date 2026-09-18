import io
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import app


class CaptureQualityTest(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def photo(self, image):
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        return self.client.post('/api/photo-quality', data={
            'image': (io.BytesIO(encoded.tobytes()), 'capture.jpg')})

    def test_corrupt_image_cannot_pass(self):
        response = self.client.post('/api/photo-quality', data={
            'image': (io.BytesIO(b'not jpeg'), 'capture.jpg')})
        self.assertEqual(response.json['status'], 'retake')

    def test_small_blurred_photo_requires_retake(self):
        data = self.photo(np.full((200, 150, 3), 220, np.uint8)).json
        self.assertEqual(data['status'], 'retake')
        self.assertFalse(data['angle_verified'])

    def test_unknown_paper_is_warning_not_false_pass(self):
        image = np.random.default_rng(4).integers(0, 255, (640, 480, 3), dtype=np.uint8)
        with patch.object(app, '_detect_reliable_paper_corners', return_value=None):
            data = self.photo(image).json
        self.assertEqual(data['status'], 'warning')
        self.assertFalse(data['paper_detected'])
        self.assertFalse(data['angle_verified'])

    def test_error_identifies_views(self):
        message = app._segmentation_retake_message([0.6, 0.95, 0.55])
        self.assertIn('正上方、右侧斜拍', message)
        self.assertNotIn('左侧斜拍', message)
        self.assertEqual(
            message.retake_slots, ['image_measure', 'image_right'])

    def test_measurement_disagreement_only_blocks_two_plausible_results(self):
        self.assertTrue(app._measurements_disagree(
            (240, 84, 55), (248, 92, 55)))
        self.assertFalse(app._measurements_disagree(
            (240, 84, 55), (243, 87, 58)))
        # 被阴影扩到超出物理范围的复核候选应被淘汰，不能反过来
        # 否决已经通过三视图几何的主结果。
        self.assertFalse(app._measurements_disagree(
            (240, 84, 55), (243, 139, 64)))

    def test_joint_failure_returns_slots_that_must_be_rephotographed(self):
        image = np.random.default_rng(8).integers(
            0, 255, (640, 480, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        error = app._segmentation_retake_message([0.5, 0.9, 0.6])
        data = {'foot_side': 'left'}
        for key in ('image_measure', 'image_left', 'image_right'):
            data[key] = (io.BytesIO(encoded.tobytes()), key + '.jpg')
        with patch.object(app, 'process_multiview_images',
                          return_value=(None, error)):
            response = self.client.post('/api/measure', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json['retake_slots'],
            ['image_measure', 'image_right'])

    def test_complete_paper_is_anchor_but_inner_white_patch_is_not(self):
        image = np.full((800, 600, 3), 70, np.uint8)
        image[120:680, 100:500] = 240
        complete = np.float32([[100, 120], [499, 120], [499, 679], [100, 679]])
        inner = np.float32([[200, 220], [499, 220], [499, 550], [200, 550]])
        self.assertIsNotNone(app._lock_complete_paper_recovery(image, complete))
        self.assertIsNone(app._lock_complete_paper_recovery(image, inner))

    def test_detected_quad_with_unsupported_side_cannot_pass(self):
        image = np.random.default_rng(5).integers(0, 255, (640, 480, 3), dtype=np.uint8)
        quad = np.float32([[60, 80], [420, 80], [420, 560], [60, 560]])
        mask = np.full((640, 480), 255, np.uint8)
        with patch.object(app, '_detect_reliable_paper_corners', return_value=quad), patch.object(
                app, '_paper_quad_evidence', return_value=5), patch.object(
                app, '_paper_quad_edge_support', return_value=([30]*4, [1, .2, .8, 1])), patch.object(
                app, '_multiview_foot_mask', return_value=mask):
            data = self.photo(image).json
        self.assertEqual(data['status'], 'warning')
        self.assertFalse(data['paper_boundary_verified'])

    def test_onnx_loaded_from_bytes(self):
        app._SEGMENTER_LOCAL.nets = {}
        net = unittest.mock.MagicMock()
        net.forward.return_value = np.ones((1, 1, 512, 384), np.float32)
        with patch.object(app.os.path, 'exists', return_value=True), patch.object(
                app.np, 'fromfile', return_value=np.array([1, 2, 3], np.uint8)), patch.object(
                app.cv2.dnn, 'readNetFromONNX', return_value=net) as loader:
            mask = app._binary_segmenter_mask(np.zeros((20, 20, 3), np.uint8), '中文/model.onnx', .35)
        self.assertIsInstance(loader.call_args.args[0], np.ndarray)
        self.assertEqual(mask.shape, (20, 20))
        app._SEGMENTER_LOCAL.nets = {}


if __name__ == '__main__':
    unittest.main()
