#!/usr/bin/env python3
"""Create local-only Streamlit secrets from an existing Google service account.

Only the local .streamlit/secrets.toml is written; never commit it to GitHub.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--sheet-id', required=True, help='Python Reports Google Sheet ID (from its URL)')
p.add_argument('--credentials', default=str(Path.home() / 'Desktop' / 'service_account.json'))
a = p.parse_args()
info = json.loads(Path(a.credentials).expanduser().read_text())
if not info.get('private_key') or not info.get('client_email'):
    raise ValueError('This is not a Google service-account credentials JSON file')
root = Path(__file__).resolve().parent
path = root / '.streamlit' / 'secrets.toml'
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w') as f:
    f.write('sheet_id = ' + json.dumps(a.sheet_id) + '\n\n[gcp_service_account]\n')
    for k, v in info.items():
        if isinstance(v, str):
            f.write(k + ' = ' + json.dumps(v) + '\n')
print('Created local private secrets:', path)
print('This file is gitignored. Copy its contents into Streamlit Cloud > App settings > Secrets.')
