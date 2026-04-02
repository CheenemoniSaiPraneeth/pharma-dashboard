"""
server.py — Serves the Pharma Intelligence dashboard on Render.

Render sets a PORT env variable. This serves:
  GET /                    → index.html
  GET /briefs_history.json → briefs_history.json  (live data)
  GET /static/*            → any file in ./static/

No dependencies beyond stdlib.
"""

import http.server
import json
import os
import socketserver
from pathlib import Path

PORT = int(os.environ.get("PORT", 8000))
BASE = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        path = self.path.split("?")[0]   # strip query string

        if path == "/" or path == "":
            self.serve_file(BASE / "index.html", "text/html")
        elif path == "/briefs_history.json":
            self.serve_file(BASE / "briefs_history.json", "application/json")
        elif path == "/briefs.json":
            self.serve_file(BASE / "briefs.json", "application/json")
        else:
            # Serve anything else from BASE (css, js, icons, etc.)
            target = BASE / path.lstrip("/")
            if target.exists() and target.is_file():
                mime = self.guess_type(str(target))
                self.serve_file(target, mime)
            else:
                self.send_error(404, "Not found")

    def serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404, f"File not found: {path.name}")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"[SERVER] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"[SERVER] Pharma Intel Dashboard running on port {PORT}")
        httpd.serve_forever()
