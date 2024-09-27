from pathlib import Path
import pharmacoforge

def fix_relative_path(path: str) -> str:
    """Given a filepath, make it relative to the root of the repository."""
    root_dir = Path(pharmacoforge.__file__).parent.parent
    return str(root_dir / path)