"""Local web server. No ROS node, GUI toolkit, or web framework is needed."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import signal
import tempfile
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid
import webbrowser

import torch

from lbks_gui.engine import PreviewEngine
from lbks_gui.robot import load_robot, load_variants


def package_paths():
    """Find source files when developing, or installed ROS package resources."""
    source = Path(__file__).resolve().parents[1]
    if (source / 'web/index.html').is_file():
        return source, source.parent / 'lbks_description'
    from ament_index_python.packages import get_package_share_directory
    return (Path(get_package_share_directory('lbks_gui')),
            Path(get_package_share_directory('lbks_description')))


def safe_file(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError('File not found')
    return path


class Studio:
    """Small, shared model cache. Each browser supplies its own target and seed."""

    def __init__(self, urdf_path, models_path, tip, assets, web):
        self.urdf, self.frame = load_robot(urdf_path)
        self.engine = PreviewEngine(self.urdf, tip)
        self.variants = load_variants(models_path)
        self.tip, self.assets, self.web = tip, Path(assets), Path(web)
        self.lock = threading.Lock()
        self.uploads = tempfile.TemporaryDirectory(prefix='lbks-checkpoints-')
        self.uploaded = {}

    def config(self):
        return {'frame': self.frame, 'tip': self.tip, 'names': self.engine.names,
                'lower': self.engine.lower.tolist(), 'upper': self.engine.upper.tolist(),
                'models': list(self.variants)}

    def checkpoint(self, filename, data):
        if Path(filename).suffix.lower() not in ('.pth', '.pt'):
            raise ValueError('Choose a .pth or .pt state dictionary')
        key = 'upload:' + uuid.uuid4().hex
        path = Path(self.uploads.name) / (key[7:] + '.pth')
        path.write_bytes(data)
        self.uploaded[key] = {'path': str(path)}
        return {'id': key, 'name': Path(filename).name}

    def run(self, operation, data):
        # Serialize model execution; the browser coalesces rapid edits.
        with self.lock:
            if operation == 'joints':
                q = torch.as_tensor(data['joints'], dtype=torch.float32)
                if q.shape != (len(self.engine.names),) or not torch.isfinite(q).all():
                    raise ValueError('Expected one finite value per joint')
                q = q.clamp(self.engine.lower, self.engine.upper)
                return {'joints': q.tolist(), 'actual': self.engine.fk(q).tolist(),
                        'network_trace': {'stages': [], 'note': 'Manual joint preview. No network prediction.'}}
            if operation != 'predict':
                raise ValueError('Unknown operation')
            name = data.get('model', 'numerical')
            if name == 'numerical':
                spec = None
            elif name == '__custom__':
                spec = {'path': str(Path(data['path']).expanduser())}
            else:
                spec = self.variants.get(name) or self.uploaded.get(name)
                if spec is None:
                    raise ValueError('Unknown model')
            seed = torch.as_tensor(data.get('seed', [0.] * len(self.engine.names)), dtype=torch.float32)
            if seed.shape != (len(self.engine.names),) or not torch.isfinite(seed).all():
                raise ValueError('Invalid seed joint vector')
            start = time.perf_counter()
            result = self.engine.solve(data['target'], seed, spec)
            result['elapsed_ms'] = (time.perf_counter() - start) * 1000
            return result

    def close(self):
        self.uploads.cleanup()


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, studio, **kwargs):
        self.studio = studio
        super().__init__(*args, **kwargs)

    def allowed(self):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:
            return False
        origin = self.headers.get('Origin')
        return origin is None or origin in {'http://' + host for host in hosts}

    def send(self, data, content_type='application/json', status=200):
        if isinstance(data, dict):
            data = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.allowed():
            self.send({'error': 'Local requests only'}, status=403)
            return
        route = unquote(urlsplit(self.path).path)
        try:
            if route == '/api/config':
                self.send(self.studio.config())
            elif route == '/robot.urdf':
                self.send(self.studio.urdf.encode(), 'application/xml')
            else:
                if route.startswith('/robot-assets/'):
                    path = safe_file(self.studio.assets, route[len('/robot-assets/'):])
                else:
                    path = safe_file(self.studio.web, 'index.html' if route == '/' else route.lstrip('/'))
                content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                self.send(path.read_bytes(), content_type)
        except (FileNotFoundError, OSError):
            self.send({'error': 'File not found'}, status=404)

    def do_POST(self):
        if not self.allowed():
            self.send({'error': 'Local requests only'}, status=403)
            return
        route = urlsplit(self.path).path
        try:
            if route not in ('/api/predict', '/api/joints', '/api/checkpoint'):
                self.send({'error': 'Unknown endpoint'}, status=404)
                return
            length = int(self.headers.get('Content-Length', '0'))
            maximum = 64 * 1024 * 1024 if route == '/api/checkpoint' else 65536
            if not 0 < length <= maximum:
                self.send({'error': 'Request is empty or too large'}, status=413)
                return
            body = self.rfile.read(length)
            if route == '/api/checkpoint':
                filename = unquote(self.headers.get('X-Filename', 'checkpoint.pth'))
                self.send(self.studio.checkpoint(filename, body))
            else:
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('Expected application/json')
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError('Expected a JSON object')
                self.send(self.studio.run(route.rsplit('/', 1)[-1], data))
        except Exception as error:
            # A failed prediction never replaces the browser's current pose.
            self.send({'error': str(error)}, status=400)

    def log_message(self, format, *args):
        if len(args) > 1 and str(args[1]) not in ('200', '304'):
            super().log_message(format, *args)


def main():
    share, description = package_paths()
    parser = argparse.ArgumentParser(description='LBKS local web GUI')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--urdf-path', default=str(description / 'urdf/ur/ur.urdf'))
    parser.add_argument('--models-config', default=str(share / 'config/models.yaml'))
    parser.add_argument('--tip-link', default='tool0')
    parser.add_argument('--assets', default=str(description))
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    studio = Studio(args.urdf_path, args.models_config, args.tip_link, args.assets, share / 'web')
    try:
        with ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, studio=studio)) as server:
            url = f'http://127.0.0.1:{server.server_port}'
            print(f'LBKS web GUI: {url}', flush=True)
            if args.open_browser:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        # ROS launch may forward a second SIGINT while Python is cleaning up.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    finally:
        studio.close()


if __name__ == '__main__':
    main()
