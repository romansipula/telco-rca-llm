"""Stream large Kaggle kernel outputs to disk (the kaggle CLI buffers whole files in RAM, which got
a 14 GB GGUF download OOM-killed on a 15 GB machine).

    python scripts/kaggle_download.py <owner/kernel> <dest_dir> [regex]
"""
import re
import subprocess
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def main():
    kernel, dest = sys.argv[1], Path(sys.argv[2])
    pat = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    owner, slug = kernel.split("/")
    api = KaggleApi()
    api.authenticate()
    token = None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            req.page_size = 50
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in resp.files:
                if pat and not pat.search(f.file_name):
                    continue
                out = dest / f.file_name
                out.parent.mkdir(parents=True, exist_ok=True)
                print("downloading", f.file_name, flush=True)
                for _ in range(20):  # resume after dropped HTTP/2 streams
                    rc = subprocess.run(["curl", "-sSL", "--retry", "5", "-C", "-", "-o", str(out), f.url]).returncode
                    if rc == 0:
                        break
                else:
                    raise SystemExit(f"download of {f.file_name} kept failing")
            token = resp.next_page_token
            if not token:
                break


if __name__ == "__main__":
    main()
