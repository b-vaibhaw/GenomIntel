import os
from cryptography.fernet import Fernet

# Determine project directories
scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(scripts_dir)
DB_FILE = os.path.join(project_dir, "biointel.db")

def get_encryption_key() -> bytes:
    """Retrieve the encryption key from .env or generate a new one if not present."""
    key = os.getenv("BIOINTEL_ENCRYPTION_KEY")
    if key:
        return key.encode('utf-8')
        
    env_path = os.path.join(project_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("BIOINTEL_ENCRYPTION_KEY="):
                    key = line.strip().split("=", 1)[1].strip()
                    break
                    
        # Fallback to AIRFLOW__CORE__FERNET_KEY if BIOINTEL_ENCRYPTION_KEY is absent
        if not key:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("AIRFLOW__CORE__FERNET_KEY="):
                        key = line.strip().split("=", 1)[1].strip()
                        break
                        
    # If still not found, generate dynamically and append to .env
    if not key:
        key = Fernet.generate_key().decode('utf-8')
        if os.path.exists(env_path):
            try:
                with open(env_path, "a", encoding="utf-8") as f:
                    f.write(f"\n# Key used for database-level user data encryption\nBIOINTEL_ENCRYPTION_KEY={key}\n")
            except Exception:
                pass
                
    os.environ["BIOINTEL_ENCRYPTION_KEY"] = key
    return key.encode('utf-8')

_fernet = None
def get_fernet():
    global _fernet
    if _fernet is None:
        key = get_encryption_key()
        _fernet = Fernet(key)
    return _fernet

def encrypt_data(plain_text: str) -> str:
    """Encrypt plain text and return a base64 encoded token string."""
    if plain_text is None:
        return None
    plain_str = str(plain_text)
    f = get_fernet()
    return f.encrypt(plain_str.encode('utf-8')).decode('utf-8')

def decrypt_data(cipher_text: str) -> str:
    """Decrypt base64 ciphertext and return the original plain text string. Falls back to plain text if not encrypted."""
    if cipher_text is None:
        return None
    cipher_str = str(cipher_text)
    f = get_fernet()
    try:
        return f.decrypt(cipher_str.encode('utf-8')).decode('utf-8')
    except Exception:
        # Fallback if the data is not encrypted (legacy DB records or plain text)
        return cipher_str
