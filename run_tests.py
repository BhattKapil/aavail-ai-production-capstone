"""Single command for the complete unit-test suite."""
import subprocess, sys
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"]))
