#!/usr/bin/env python3
"""Start the validated local runtime and chat with the installed model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs/native-runtime-v8"
URL = "http://127.0.0.1:11435"


def api(path):
    with urllib.request.urlopen(URL + path, timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("prompt", nargs="*")
    args = parser.parse_args()
    release = json.loads((RUNTIME / "release.json").read_text())
    binary = RUNTIME / "bin/ollama"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != release["binary_sha256"]:
        raise SystemExit("Runtime binary does not match the validated release")
    env = os.environ.copy()
    env.update(OLLAMA_HOST=URL, OLLAMA_MODELS=str(ROOT / "outputs/native-model-store-v1"),
               OLLAMA_NO_CLOUD="1", OLLAMA_MAX_LOADED_MODELS="1",
               OLLAMA_NUM_PARALLEL="1", OLLAMA_CONTEXT_LENGTH="4096")
    try:
        version = api("/api/version")
    except urllib.error.URLError:
        if args.status:
            raise SystemExit("Native server is stopped")
        (ROOT / "runs").mkdir(exist_ok=True)
        with (ROOT / "runs/native-server-launcher.log").open("ab") as log:
            process = subprocess.Popen([str(binary), "serve"], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        for _ in range(60):
            if process.poll() is not None:
                raise SystemExit("Native server failed; inspect runs/native-server-launcher.log")
            try:
                version = api("/api/version")
                break
            except urllib.error.URLError:
                time.sleep(1)
        else:
            raise SystemExit("Native server did not become ready")
    if version.get("version") != release["runtime_version"]:
        raise SystemExit("Port 11435 is serving a different runtime")
    models = api("/api/tags")["models"]
    matching = [row for row in models if row.get("name") == release["model"]]
    if len(matching) != 1 or matching[0].get("digest") != release["model_digest"]:
        raise SystemExit("Installed model does not match the validated release")
    if args.status:
        print(json.dumps({"url": URL, **release}, indent=2))
        return
    os.execve(str(binary), [str(binary), "run", release["model"], "--think=false", *args.prompt], env)


if __name__ == "__main__":
    main()
