import os
import json
import tempfile
import shutil
import re
import contextlib
import coverage
import pytest
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional


class ExitCode(Enum):
    """Standardized exit codes for test results."""
    OK = 0
    TESTS_FAILED = 1
    INTERRUPTED = 2
    INTERNAL_ERROR = 3
    USAGE_ERROR = 4


class AbstractTester(ABC):
    """Abstract base class for language-specific test executors."""

    def __init__(self, tmp_path: str) -> None:
        self.tmp_path = tmp_path
        os.makedirs(self.tmp_path, exist_ok=True)

    @abstractmethod
    def make_exec_file(self, code: str) -> str:
        pass

    @abstractmethod
    def make_test_file(self, tests: str) -> str:
        pass

    @abstractmethod
    def run_coverage(self, test_path: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute(self, code: str, tests: str) -> Dict[str, Any]:
        pass


class PythonTester(AbstractTester):
    """Python implementation of the tester with pytest + coverage."""

    def _extract_feedback(self, report_text: str) -> str:
        feedback = ""
        for section in ("FAILURES", "ERRORS"):
            pattern = re.compile(rf"(=+ {section} =+\n(?:.*\n)+?)(=+)", re.MULTILINE)
            match = pattern.search(report_text)
            if match:
                feedback += match.group(1)
        return feedback.strip()

    def _map_exit_code(self, code: pytest.ExitCode) -> ExitCode:
        mapping = {
            pytest.ExitCode.OK: ExitCode.OK,
            pytest.ExitCode.TESTS_FAILED: ExitCode.TESTS_FAILED,
            pytest.ExitCode.INTERRUPTED: ExitCode.INTERRUPTED,
            pytest.ExitCode.INTERNAL_ERROR: ExitCode.INTERNAL_ERROR,
            pytest.ExitCode.USAGE_ERROR: ExitCode.USAGE_ERROR,
        }
        return mapping.get(code, ExitCode.INTERNAL_ERROR)

    def _create_init_file(self) -> None:
        """Make package importable."""
        with open(os.path.join(self.tmp_path, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("")

    def make_exec_file(self, code: str) -> str:
        path = os.path.join(self.tmp_path, "file.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path

    def make_test_file(self, tests: str) -> str:
        path = os.path.join(self.tmp_path, "tests.py")
        content = f"from .file import *\n{tests}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def run_coverage(self, test_path: str) -> Dict[str, Any]:
        cov = coverage.Coverage(source=[self.tmp_path])
        cov.start()

        stdout_path = os.path.join(self.tmp_path, "stdout.txt")
        stderr_path = os.path.join(self.tmp_path, "stderr.txt")

        with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            result_code = pytest.main([test_path, "--tb=short"])

        cov.stop()
        cov.save()

        report_path = os.path.join(self.tmp_path, "coverage.json")
        cov.json_report(outfile=report_path, pretty_print=True)

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        key = f"{self.tmp_path}/file.py"
        coverage_percent = report["files"].get(key, {}).get("summary", {}).get("percent_covered", 0)

        with open(stdout_path, "r", encoding="utf-8") as f:
            feedback_text = f.read()

        return {
            "coverage": coverage_percent,
            "exit_code": self._map_exit_code(result_code),
            "feedback": self._extract_feedback(feedback_text),
        }

    def execute(self, code: str, tests: str) -> Dict[str, Any]:
        """Run the complete workflow for Python code and tests."""
        self._create_init_file()
        self.make_exec_file(code)
        test_path = self.make_test_file(tests)
        return self.run_coverage(test_path)


class Tester:
    """
    Master tester that automatically manages temporary folders.
    Usage is simple:
        tester = Tester()
        result = tester.run("python", code, tests)
    """

    def __init__(self, testers: Optional[Dict[str, type]] = None) -> None:
        self.testers = testers or {"python": PythonTester}

    def run(self, language: str, code: str, tests: str) -> Dict[str, Any]:
        """Run tests for a given language inside an automatic temp folder."""
        if language not in self.testers:
            raise ValueError(f"Unsupported language: {language}")

        # create and clean up temporary workspace automatically
        with tempfile.TemporaryDirectory() as tmp_dir:
            tester_cls = self.testers[language]
            tester = tester_cls(tmp_dir)
            return tester.execute(code, tests)
