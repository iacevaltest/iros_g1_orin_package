#!/usr/bin/env python3
"""Live MJPEG relay of the organizer's camera stream (:5555, real_orin.py) --
view any camera in a browser, from any device on the local network, with no
display needed on either PC2 or wherever this relay itself runs.

READ-ONLY: one SUB socket into :5555, publishes nothing back to the robot
side. Safe to run any time, alongside anything else.

    python3 camera_live_view.py --camera-host $ROBOT_HOST --port 8081

Then open http://<this-machine's-ip>:8081/ in a browser on any device on the
same network as the robot -- shows every camera key currently being
published (ego_view, ego_view_left/right if PUBLISH_STEREO is on,
left_wrist, right_wrist), each as a live-updating image.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import msgpack
import zmq

_lock = threading.Lock()
_latest_jpegs: dict[str, bytes] = {}


def camera_subscriber(host: str, port: int):
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.CONFLATE, 1)   # always the newest frame, never a backlog
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    print(f"[camera_live_view] subscribed to tcp://{host}:{port}")
    while True:
        try:
            blob = sock.recv()
            msg = msgpack.unpackb(blob, raw=False)
            images = msg.get("images", {})
            with _lock:
                for key, jpg in images.items():
                    _latest_jpegs[key] = jpg
        except Exception as exc:
            print(f"[camera_live_view] decode error: {exc}")
            time.sleep(0.1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet -- an access log line per MJPEG chunk would be noise

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with _lock:
                keys = sorted(_latest_jpegs.keys())
            body = "<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
            body += "<h3>Live camera views</h3>"
            if not keys:
                body += "<p>No camera frames received yet -- is real_orin.py running?</p>"
            for k in keys:
                body += f"<h4>{k}</h4><img src='/stream/{k}' width='640'><br>"
            body += "</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body.encode())
            return

        if self.path.startswith("/stream/"):
            key = self.path[len("/stream/"):]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        jpg = _latest_jpegs.get(key)
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404)
        self.end_headers()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera-host", default=os.environ.get("ROBOT_HOST", "127.0.0.1"),
                    help="Robot onboard computer IP -- where real_orin.py publishes :5555 "
                         "(or set ROBOT_HOST)")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument("--port", type=int, default=8081, help="local HTTP port to serve on")
    p.add_argument("--bind", default="0.0.0.0")
    args = p.parse_args()

    threading.Thread(target=camera_subscriber, args=(args.camera_host, args.camera_port),
                      daemon=True).start()

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"[camera_live_view] open http://<this-machine-ip>:{args.port}/ in a browser")
    server.serve_forever()


if __name__ == "__main__":
    main()
