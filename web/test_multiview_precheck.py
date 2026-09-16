import unittest

import cv2
import numpy as np

import app


class MultiviewPrecheckTest(unittest.TestCase):
    def setUp(self):
        y, x = np.indices((720, 960))
        self.image = np.dstack((
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ))

    def test_exact_duplicate_is_rejected(self):
        self.assertEqual(
            app._duplicate_multiview_pair([
                self.image, self.image.copy(), np.flipud(self.image)]),
            (0, 1))

    def test_jpeg_reencode_is_rejected(self):
        ok, encoded = cv2.imencode(
            '.jpg', self.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.assertTrue(ok)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        self.assertEqual(
            app._duplicate_multiview_pair([
                self.image, np.fliplr(self.image), recompressed]),
            (0, 2))

    def test_different_views_are_accepted(self):
        self.assertIsNone(app._duplicate_multiview_pair([
            self.image, np.fliplr(self.image), np.flipud(self.image)]))


if __name__ == '__main__':
    unittest.main()
