import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXEC_TIMEOUT = 30
MAX_OUTPUT = 8000

_PREAMBLE = """\
import os, warnings
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings('ignore')

try:
    import matplotlib.style as _mplstyle
    _orig_style_use = _mplstyle.use
    def _safe_style_use(style, *a, **kw):
        try:
            _orig_style_use(style, *a, **kw)
        except Exception:
            pass
    _mplstyle.use = _safe_style_use
except Exception:
    pass

try:
    import matplotlib.pyplot as _plt_patch
    _plt_patch.show = lambda *a, **kw: None
except Exception:
    pass
"""

_POSTAMBLE = """
try:
    import matplotlib.pyplot as _plt, base64 as _b64, io as _io
    _imgs = []
    for _n in _plt.get_fignums():
        _buf = _io.BytesIO()
        _plt.figure(_n).savefig(_buf, format='png', dpi=100, bbox_inches='tight')
        _plt.close(_n)
        _imgs.append(_b64.b64encode(_buf.getvalue()).decode())
    if _imgs:
        print('\\n__NEXUS_IMGS__' + '||'.join(_imgs) + '__END_IMGS__')
except Exception:
    pass
"""


def execute_python(code: str, timeout: int = EXEC_TIMEOUT) -> dict:
    work_dir = Path(tempfile.mkdtemp(prefix="nexus_exec_"))
    script_path = work_dir / "run.py"
    script_path.write_text(_PREAMBLE + code + "\n" + _POSTAMBLE, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        images: list[str] = []
        if "__NEXUS_IMGS__" in stdout:
            before, rest = stdout.split("__NEXUS_IMGS__", 1)
            img_part, after = rest.split("__END_IMGS__", 1)
            stdout = before + after
            images = [x for x in img_part.split("||") if x]
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout.strip()[:MAX_OUTPUT],
            "stderr": stderr.strip()[:MAX_OUTPUT],
            "images": images,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Tempo limite de {timeout}s excedido.",
            "images": [],
        }
    except Exception as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc), "images": []}
    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)
