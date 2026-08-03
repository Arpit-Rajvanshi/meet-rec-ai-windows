"""
Unit Tests for CacheResolver Component.
Tests environment detection, configuration precedence, cross-platform path resolution,
strategy selection, override handling, and normalization.
Zero top-level dependencies on SpeechBrain.
"""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from cache_resolver import CacheResolver


class TestCacheResolver(unittest.TestCase):

    def setUp(self):
        # Save original environment
        self._orig_env = dict(os.environ)

    def tearDown(self):
        # Restore environment
        os.environ.clear()
        os.environ.update(self._orig_env)

    def test_environment_detection_runpod(self):
        os.environ["RUNPOD_POD_ID"] = "pod-12345"
        self.assertEqual(CacheResolver.detect_environment(), "runpod")

    def test_environment_detection_ci(self):
        os.environ.pop("RUNPOD_POD_ID", None)
        os.environ["CI"] = "true"
        self.assertEqual(CacheResolver.detect_environment(), "ci")

    def test_windows_path_resolution_ignores_linux_volume(self):
        """Verify Windows environment ignores /runpod-volume Linux paths in SPEECHBRAIN_CACHE."""
        os.environ.pop("RUNPOD_POD_ID", None)
        os.environ.pop("CI", None)
        os.environ["SPEECHBRAIN_CACHE"] = "/runpod-volume/speechbrain/spkrec-ecapa-voxceleb"
        
        with patch("sys.platform", "win32"), patch("os.name", "nt"):
            path = CacheResolver.get_speechbrain_cache()
            self.assertNotIn("runpod-volume", path)
            self.assertTrue(os.path.isabs(path))
            self.assertTrue(path.endswith("spkrec-ecapa-voxceleb"))

    def test_runpod_path_resolution(self):
        """Verify RunPod environment preserves /runpod-volume persistent path."""
        os.environ["RUNPOD_POD_ID"] = "pod-test"
        os.environ["SPEECHBRAIN_CACHE"] = "/runpod-volume/speechbrain/spkrec-ecapa-voxceleb"
        
        with patch.object(CacheResolver, "detect_environment", return_value="runpod"):
            path = CacheResolver.get_speechbrain_cache()
            self.assertIn("runpod-volume", path)

    def test_strategy_selection_per_environment(self):
        """Verify strategy selection returns 'copy' for Windows and 'symlink' for Linux/RunPod/Docker/CI."""
        with patch.object(CacheResolver, "detect_environment", return_value="windows"):
            self.assertEqual(CacheResolver.get_speechbrain_local_strategy(), "copy")

        for env in ["linux", "runpod", "docker", "ci"]:
            with patch.object(CacheResolver, "detect_environment", return_value=env):
                self.assertEqual(CacheResolver.get_speechbrain_local_strategy(), "symlink")

    def test_custom_override_precedence(self):
        """Explicit override_path should take highest precedence over environment variables."""
        os.environ["SPEECHBRAIN_CACHE"] = "/some/env/path"
        override = os.path.abspath("./custom_override_dir")
        
        path = CacheResolver.get_speechbrain_cache(override_path=override)
        self.assertEqual(path, override)

    def test_path_normalization_and_creation(self):
        """Path should be returned as absolute and directory created."""
        target_dir = os.path.abspath("./tmp_test_cache_dir")
        path = CacheResolver._normalize_and_create(target_dir)
        
        self.assertEqual(path, target_dir)
        self.assertTrue(os.path.exists(target_dir))
        
        # Cleanup
        if os.path.exists(target_dir):
            try:
                os.rmdir(target_dir)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
