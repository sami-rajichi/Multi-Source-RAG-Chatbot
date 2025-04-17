import os
import shutil
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def force_delete_directory(path: str) -> bool:
    """Forcefully delete a directory and all its contents, handling file locks on Windows."""
    if not os.path.exists(path):
        return True
    
    # First try normal deletion
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        logger.warning(f"Normal deletion failed, attempting forceful deletion: {e}")
    
    # If normal fails, try forceful deletion
    try:
        # Windows-specific solution
        if os.name == 'nt':
            try:
                handle_path = os.path.join(os.environ["SystemRoot"], "System32", "handle.exe")
                if os.path.exists(handle_path):
                    os.system(f'{handle_path} -accepteula {path} > nul 2>&1')
                    os.system(f'{handle_path} -accepteula -p {path} > nul 2>&1')
            except:
                pass
            
            os.system(f'rd /s /q "{path}"')
        else:  # Unix-like
            os.system(f'rm -rf "{path}"')
        
        return not os.path.exists(path)
    except Exception as e:
        logger.error(f"Forceful deletion failed: {e}")
        return False

def delete_data_files(data_path: str) -> Tuple[int, List[str]]:
    """Delete all files in data directory."""
    deleted_count = 0
    failed_files = []
    
    if os.path.exists(data_path) and os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path): 
                    os.unlink(file_path)
                    deleted_count += 1
            except Exception as e: 
                logger.error(f"Failed to delete {file_path}: {e}") 
                failed_files.append(filename)
    
    return deleted_count, failed_files

def count_data_files(data_path: str) -> int:
    """Count files in data directory."""
    if not os.path.exists(data_path) or not os.path.isdir(data_path):
        return 0
    return len([entry for entry in os.scandir(data_path) if entry.is_file()])

def safe_filename(original_name: str) -> str:
    """Create a safe filename by removing special characters."""
    return "".join(c for c in original_name if c.isalnum() or c in (' ','.','_')).rstrip()