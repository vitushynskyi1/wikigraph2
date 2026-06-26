import os
import gzip
import shutil
import urllib.request
from pathlib import Path

DUMP_DIR = Path(__file__).resolve().parent / "dumps"
DUMP_DIR.mkdir(exist_ok=True)
BASE_URL = "https://dumps.wikimedia.org/skwiki/latest/"

FILES = {
    "page": ("skwiki-latest-page.sql", "pages"),
    "linktarget": ("skwiki-latest-linktarget.sql", "linktargets"),
    "pagelinks": ("skwiki-latest-pagelinks.sql", "links")
}

for target, (sql_name, alias) in FILES.items():
    sql_path = DUMP_DIR / sql_name
    link_path = DUMP_DIR / alias
    url = f"{BASE_URL}skwiki-latest-{target}.sql.gz"
    
    print(f"Processing {target}...")
    
    with urllib.request.urlopen(url) as response:
        with gzip.open(response, "rb") as f_in, open(sql_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    if link_path.exists(): 
        link_path.unlink()
    os.link(sql_path, link_path)

print("Done. All dumps configured.")
