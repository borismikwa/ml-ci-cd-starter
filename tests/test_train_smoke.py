import subprocess
import sys


def test_train_script_runs():
    r = subprocess.run(
        [sys.executable, "src/train.py", "--config", "configs/train_config.yml"], check=False
    )
    assert r.returncode == 0
