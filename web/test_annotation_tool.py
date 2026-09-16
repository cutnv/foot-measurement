import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate_feet


class AnnotationToolTest(unittest.TestCase):
    def test_save_reviewed_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "photos"
            folder = source / "one"
            folder.mkdir(parents=True)
            image = np.full((80, 60, 3), 255, np.uint8)
            cv2.imwrite(str(folder / "a.png"), image)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{
                "group": 1, "folder": "one",
                "input_order": ["a.png"],
            }]), "utf-8")
            server = annotate_feet.create_app(
                source, root / "labels", manifest, 1)
            client = server.test_client()
            self.assertEqual(client.get("/api/items").status_code, 200)
            page = client.get("/").get_data(as_text=True)
            token = page.split("const token=", 1)[1].split(", labels=", 1)[0]
            token = json.loads(token)
            ok, encoded = cv2.imencode(
                ".png", np.full((80, 60), 255, np.uint8))
            response = client.post("/api/items/0", data={
                "mask": (io.BytesIO(encoded.tobytes()), "mask.png"),
                "role": "top",
                "points": json.dumps({"longest_toe": [10, 20]}),
            }, headers={"X-Annotation-Token": token})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(client.get("/api/items").get_json()[0]["reviewed"])


if __name__ == "__main__":
    unittest.main()
