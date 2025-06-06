import requests

# ─── REPLACE THESE ──────────────────────────────────────────────────────────────
USERNAME       = "abreg2025"        # your Mapbox username
MASTER_TOKEN   = "sk.eyJ1IjoiYWJyZWcyMDI1IiwiYSI6ImNtYmtmcXl2dzA5bW4yaXB4aWt2bmN6cGsifQ.qHegAz9uKXpDaMwQZN8SWw"  # token that can create new tokens
# ───────────────────────────────────────────────────────────────────────────────

# The scopes we actually want in our new token:
new_scopes = [
    "tilesets:write",
    "tilesets:read",
    "sources:write",
    "sources:read"
]

url = f"https://api.mapbox.com/tokens/v2/{USERNAME}?access_token={MASTER_TOKEN}"
payload = {
    "note":   "Demo token with MTS & source scopes",
    "scopes": new_scopes
}

resp = requests.post(url, json=payload)
if resp.status_code in (200, 201):
    data = resp.json()
    print("✔︎ Created new token successfully!\n")
    print("  → token:", data["token"])
    print("  → scopes:", data["scopes"])
    print("  → id:", data["id"])
else:
    print("✘ Failed to create token:", resp.status_code)
    print(resp.text)
