"""
Cross-Platform Cache Resolver for Machine Learning Models (SpeechBrain, PyAnnote, HuggingFace).
Handles environment-aware path resolution and fetching strategy selection across Windows, Linux, RunPod, Docker, and CI.
Zero top-level dependencies on SpeechBrain to prevent import side-effects during early pipeline stages.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger("cache_resolver")


class CacheResolver:
    """
    Centralized cross-platform cache path and strategy resolver.
    
    Precedence Order:
    1. Explicit runtime parameter (override_path)
    2. Environment variable (e.g., SPEECHBRAIN_CACHE, HF_HOME) IF valid for OS
    3. Platform / Environment defaults:
       - RunPod: /runpod-volume/... (persistent volume)
       - Docker: <workdir>/pretrained_models/...
       - Windows: <workdir>/pretrained_models/... (prevents cross-drive NTFS symlink WinError 1314)
       - Linux / CI: <workdir>/pretrained_models/... or ~/.cache/...
    """
    
    @staticmethod
    def detect_environment() -> str:
        """Detect execution environment: 'runpod', 'docker', 'ci', 'windows', or 'linux'."""
        if os.getenv("RUNPOD_POD_ID"):
            return "runpod"
        if os.path.exists("/.dockerenv"):
            return "docker"
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            return "ci"
        if sys.platform == "win32" or os.name == "nt":
            return "windows"
        return "linux"

    @classmethod
    def get_speechbrain_local_strategy(cls) -> str:
        """
        Return appropriate strategy string ('copy' or 'symlink') for the current environment.
        - Windows: 'copy' (avoids NTFS symlink privilege error WinError 1314)
        - Linux / RunPod / Docker / CI: 'symlink'
        """
        env_type = cls.detect_environment()
        if env_type == "windows":
            return "copy"
        return "symlink"

    @classmethod
    def resolve_speechbrain_strategy_object(cls, strategy_name: Optional[str] = None) -> Any:
        """
        Lazy resolution: Imports SpeechBrain ONLY when invoked in Stage 4,
        converting string 'copy'/'symlink' to speechbrain.utils.fetching.LocalStrategy.
        """
        strat_str = strategy_name or cls.get_speechbrain_local_strategy()
        try:
            from speechbrain.utils.fetching import LocalStrategy
            if strat_str == "copy":
                return LocalStrategy.COPY
            return LocalStrategy.SYMLINK
        except ImportError:
            return strat_str

    @classmethod
    def get_speechbrain_cache(cls, override_path: Optional[str] = None, workdir: Optional[str] = None) -> str:
        """
        Resolve absolute path for SpeechBrain ECAPA-TDNN model cache.
        """
        env_type = cls.detect_environment()
        base_dir = Path(workdir).resolve() if workdir else Path(__file__).parent.resolve()

        # 1. Explicit runtime override
        if override_path:
            return cls._normalize_and_create(override_path)
            
        # 2. Environment variable
        env_val = os.getenv("SPEECHBRAIN_CACHE")
        if env_val:
            # On Windows, if env_val is a Linux volume path (e.g. /runpod-volume/...), ignore it
            if env_type == "windows" and env_val.startswith("/runpod-volume"):
                logger.info(f"CacheResolver [Windows]: Ignoring Linux volume env path '{env_val}'. Using local directory.")
            else:
                return cls._normalize_and_create(env_val)

        # 3. Environment-specific platform defaults
        if env_type == "runpod":
            path = Path("/runpod-volume/speechbrain/spkrec-ecapa-voxceleb")
        elif env_type == "windows":
            # Repository-local folder to ensure same-drive access and avoid cross-drive symlink WinError 1314
            path = base_dir / "pretrained_models" / "spkrec-ecapa-voxceleb"
        else: # linux, docker, ci
            path = base_dir / "pretrained_models" / "spkrec-ecapa-voxceleb"

        return cls._normalize_and_create(str(path))

    @classmethod
    def get_huggingface_cache(cls, override_path: Optional[str] = None) -> str:
        """
        Resolve absolute path for HuggingFace Hub model cache.
        """
        env_type = cls.detect_environment()
        
        if override_path:
            return cls._normalize_and_create(override_path)
            
        env_val = os.getenv("HF_HOME")
        if env_val:
            if env_type == "windows" and env_val.startswith("/runpod-volume"):
                logger.info(f"CacheResolver [Windows]: Ignoring Linux volume HF_HOME path '{env_val}'. Using default.")
            else:
                return cls._normalize_and_create(env_val)
                
        if env_type == "runpod":
            return cls._normalize_and_create("/runpod-volume/huggingface")
            
        default_hf = Path.home() / ".cache" / "huggingface"
        return cls._normalize_and_create(str(default_hf))

    @classmethod
    def get_torch_cache(cls, override_path: Optional[str] = None) -> str:
        """
        Resolve absolute path for PyTorch model cache.
        """
        env_type = cls.detect_environment()
        if override_path:
            return cls._normalize_and_create(override_path)
        env_val = os.getenv("TORCH_HOME")
        if env_val:
            if env_type == "windows" and env_val.startswith("/runpod-volume"):
                pass
            else:
                return cls._normalize_and_create(env_val)
        if env_type == "runpod":
            return cls._normalize_and_create("/runpod-volume/torch")
        default_torch = Path.home() / ".cache" / "torch"
        return cls._normalize_and_create(str(default_torch))

    @classmethod
    def _normalize_and_create(cls, path_str: str) -> str:
        """Normalize path to absolute string and create directory if non-existent."""
        try:
            abs_path = os.path.abspath(os.path.expanduser(path_str))
            os.makedirs(abs_path, exist_ok=True)
            return abs_path
        except Exception as e:
            logger.error(f"Failed to create cache path '{path_str}': {e}")
            raise RuntimeError(
                f"Cache Path Resolution Error:\n"
                f"  Target Path: {path_str}\n"
                f"  Execution Environment: {cls.detect_environment()}\n"
                f"  Platform: {sys.platform}\n"
                f"  Reason: {e}\n"
                f"  Recommended Fix: Check directory write permissions or specify a valid local SPEECHBRAIN_CACHE."
            ) from e
