"""Exercise HTTP routes without opening a network socket."""

from io import BytesIO
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import torch

from lbks_gui.engine import PositionMLP
from lbks_gui.server import Handler, Studio, package_paths, safe_file


class MemorySocket:
    def __init__(self, request):
        self.input = BytesIO(request)
        self.output = BytesIO()

    def makefile(self, *args):
        return self.input

    def sendall(self, data):
        self.output.write(data)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.share, cls.description = package_paths()
        cls.studio = Studio(cls.description / 'urdf/ur/ur.urdf',
                            cls.share / 'config/models.yaml', 'tool0',
                            cls.description, cls.share / 'web')

    @classmethod
    def tearDownClass(cls):
        cls.studio.close()

    def http(self, method, path, data=None, headers=None):
        body = data if isinstance(data, bytes) else json.dumps(data).encode() if data is not None else b''
        fields = {'Host': 'localhost:8080', 'Content-Type': 'application/json',
                  'Content-Length': str(len(body)), **(headers or {})}
        request = f'{method} {path} HTTP/1.0\r\n' + ''.join(f'{k}: {v}\r\n' for k, v in fields.items())
        socket = MemorySocket(request.encode() + b'\r\n' + body)
        Handler(socket, ('127.0.0.1', 1234), SimpleNamespace(server_port=8080), studio=self.studio)
        response_headers, response = socket.output.getvalue().split(b'\r\n\r\n', 1)
        code = int(response_headers.split()[1])
        if b'Content-Type: application/json' in response_headers:
            response = json.loads(response)
        return code, response

    def test_page_config_and_robot_assets(self):
        code, page = self.http('GET', '/')
        self.assertEqual(code, 200)
        self.assertIn(b'Follow the forward pass.', page)
        code, config = self.http('GET', '/api/config')
        self.assertEqual(code, 200)
        self.assertEqual(len(config['names']), 6)
        self.assertEqual(self.http('GET', '/robot.urdf')[0], 200)
        self.assertEqual(self.http('GET', '/robot-assets/meshes/ur16e/visual/base.dae')[0], 200)

    def test_numerical_prediction_and_joint_preview(self):
        code, result = self.http('POST', '/api/predict', {'target': [.4, 0., .4], 'model': 'numerical'})
        self.assertEqual(code, 200)
        self.assertLess(result['error'], .002)
        code, manual = self.http('POST', '/api/joints', {'joints': result['joints']})
        self.assertEqual(code, 200)
        self.assertEqual(manual['actual'], result['actual'])
        self.assertEqual(manual['network_trace']['stages'], [])

    def test_upload_predict_and_network_trace(self):
        stream = BytesIO()
        torch.save(PositionMLP(6).state_dict(), stream)
        code, upload = self.http('POST', '/api/checkpoint', stream.getvalue(),
                                 {'Content-Type': 'application/octet-stream', 'X-Filename': 'test.pth'})
        self.assertEqual(code, 200)
        code, result = self.http('POST', '/api/predict', {'target': [.4, .1, .4], 'model': upload['id']})
        self.assertEqual(code, 200)
        self.assertEqual(len(result['network_trace']['stages']), 10)
        self.assertEqual(result['network_trace']['stages'][-2]['values'], result['joints'])
        self.assertEqual(result['network_trace']['stages'][-1]['values'], result['actual'])

    def test_bad_inputs_report_errors(self):
        for payload in ({'target': [1, 2]}, {'target': [float('nan'), 0, 0]},
                        {'target': [0, 0, 0], 'seed': [1, 2]},
                        {'target': [0, 0, 0], 'model': 'missing'}):
            self.assertEqual(self.http('POST', '/api/predict', payload)[0], 400)
        self.assertEqual(self.http('POST', '/api/joints', {'joints': [0]})[0], 400)
        self.assertEqual(self.http('POST', '/api/predict', b'{')[0], 400)
        self.assertEqual(self.http('POST', '/api/checkpoint', b'bad', {'X-Filename': 'test.exe'})[0], 400)
        self.assertEqual(self.http('POST', '/api/predict', {}, {'Content-Length': '99999999'})[0], 413)

    def test_local_origin_and_asset_boundaries(self):
        self.assertEqual(self.http('GET', '/', headers={'Host': 'example.com:8080'})[0], 403)
        self.assertEqual(self.http('POST', '/api/joints', {'joints': [0.] * 6},
                                   {'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.http('GET', '/robot-assets/%2e%2e/lbks_gui/setup.py')[0], 404)
        self.assertEqual(self.http('GET', '/%2e%2e/setup.py')[0], 404)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'web'
            root.mkdir()
            (Path(temp) / 'secret').write_text('private')
            (root / 'link').symlink_to(Path(temp) / 'secret')
            with self.assertRaises(FileNotFoundError):
                safe_file(root, 'link')

    def test_browser_imports_are_bundled(self):
        web = self.share / 'web'
        for file in web.rglob('*.js'):
            for dependency in re.findall(r'''(?:\bfrom\s*|\bimport\s*)['"]([^'"]+)['"]''', file.read_text()):
                if dependency == 'three':
                    target = web / 'vendor/three/build/three.module.js'
                elif dependency.startswith('three/'):
                    target = web / 'vendor/three' / dependency[6:].replace('addons/', 'examples/jsm/', 1)
                elif dependency.startswith('.'):
                    target = file.parent / dependency
                else:
                    self.fail(f'Unexpected external import: {dependency}')
                self.assertTrue(target.is_file(), f'{file}: missing {dependency}')


if __name__ == '__main__':
    unittest.main()
