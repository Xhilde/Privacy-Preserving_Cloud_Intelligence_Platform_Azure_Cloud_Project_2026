import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

CLOUD_METADATA_IPS = ["169.254.169.254", "100.100.100.200", "metadata.google.internal"]
PRIVATE_RANGES = ["10.0.0.", "172.16.0.", "192.168.1.", "192.168.0."]
EXTERNAL_DOMAINS = ["api.partner.com", "cdn.example.com", "public-api.io", "images.example.net"]
STEALTH_DOMAINS = ["cdn-cache.rebind-test.io", "asset-mirror.io", "img-proxy-service.net"]
PARSER_CONFUSION_TARGETS = ["evil-cdn.net", "attacker-controlled.io", "fake-partner-api.com"]
SCHEMES_BENIGN = ["https", "http"]
SCHEMES_MALICIOUS = ["gopher", "file", "dict", "http"]

def generate_benign_request():
    uses_literal = 1 if random.random() < 0.08 else 0
    return {
        "target_host": random.choice(EXTERNAL_DOMAINS) if not uses_literal else "10.0.5." + str(random.randint(1, 50)),
        "scheme": random.choice(SCHEMES_BENIGN),
        "uses_ip_literal": uses_literal,
        "ip_is_private": 0,
        "ip_is_metadata_endpoint": 0,
        "contains_encoding_obfuscation": 0,
        "redirect_count": random.randint(0, 2),
        "contains_credentials_in_url": 1 if random.random() < 0.03 else 0,
        "port": random.choice([80, 443, 443, 443, 8080]),
        "is_ssrf": 0
    }

def generate_ssrf_request():
    target_type = random.choice([
        "metadata", "private", "loopback", "obfuscated", "stealthy",
        "ipv6_loopback", "parser_confusion"
    ])

    if target_type == "metadata":
        host = random.choice(CLOUD_METADATA_IPS)
        is_metadata, is_private, uses_literal, scheme = 1, 0, 1, random.choice(SCHEMES_MALICIOUS)
        obfuscated, creds, redirects, port = 0, 0, random.randint(0, 3), random.choice([80, 443])

    elif target_type == "private":
        host = random.choice(PRIVATE_RANGES) + str(random.randint(1, 254))
        is_metadata, is_private, uses_literal, scheme = 0, 1, 1, random.choice(SCHEMES_MALICIOUS)
        obfuscated, creds, redirects, port = 0, 0, random.randint(0, 3), random.choice([8080, 22, 6379])

    elif target_type == "loopback":
        host = "127.0.0.1"
        is_metadata, is_private, uses_literal, scheme = 0, 1, 1, random.choice(SCHEMES_MALICIOUS)
        obfuscated, creds, redirects, port = 0, 0, random.randint(0, 2), random.choice([80, 8080])

    elif target_type == "ipv6_loopback":
        # IPv6 loopback ([::1]) is a documented SSRF bypass -- some allowlist filters
        # only check for the IPv4 form "127.0.0.1" and miss this entirely
        host = "[::1]"
        is_metadata, is_private, uses_literal, scheme = 0, 1, 1, random.choice(SCHEMES_MALICIOUS)
        obfuscated, creds, redirects, port = 0, 0, random.randint(0, 2), random.choice([80, 8080])

    elif target_type == "obfuscated":
        host = random.choice(["2130706433", "0x7f000001", "017700000001"])
        is_metadata, is_private, uses_literal, scheme = 0, 1, 1, random.choice(SCHEMES_MALICIOUS)
        obfuscated, creds, redirects, port = 1, 0, random.randint(0, 3), random.choice([80, 443])

    elif target_type == "parser_confusion":
        # http://127.0.0.1@evil.com -- some URL parsers/libraries treat "127.0.0.1" as
        # userinfo and route to evil.com; others route to 127.0.0.1 -- this ambiguity
        # itself is the documented bypass (PayloadsAllTheThings, "URL parser confusion")
        host = "127.0.0.1@" + random.choice(PARSER_CONFUSION_TARGETS)
        is_metadata, is_private, uses_literal, scheme = 0, 1, 1, "http"
        obfuscated, creds, redirects, port = 0, 1, random.randint(0, 2), 80

    else:  # stealthy
        host = random.choice(STEALTH_DOMAINS)
        is_metadata, is_private, uses_literal, scheme = 0, 0, 0, "https"
        obfuscated, creds, redirects, port = 0, 0, random.randint(1, 4), 443

    return {
        "target_host": host,
        "scheme": scheme,
        "uses_ip_literal": uses_literal,
        "ip_is_private": is_private,
        "ip_is_metadata_endpoint": is_metadata,
        "contains_encoding_obfuscation": obfuscated,
        "redirect_count": redirects,
        "contains_credentials_in_url": creds,
        "port": port,
        "is_ssrf": 1
    }

def generate_org_dataset(org_name, n_benign, n_malicious):
    rows = [generate_benign_request() for _ in range(n_benign)]
    rows += [generate_ssrf_request() for _ in range(n_malicious)]
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df["org"] = org_name
    return df

if __name__ == "__main__":
    org_a = generate_org_dataset("org-a", n_benign=2000, n_malicious=60)
    org_b = generate_org_dataset("org-b", n_benign=3500, n_malicious=40)
    org_c = generate_org_dataset("org-c", n_benign=1500, n_malicious=90)

    org_a.to_csv("dataset/raw/org_a_requests.csv", index=False)
    org_b.to_csv("dataset/raw/org_b_requests.csv", index=False)
    org_c.to_csv("dataset/raw/org_c_requests.csv", index=False)

    print(f"org-a: {len(org_a)} rows, {org_a['is_ssrf'].sum()} SSRF")
    print(f"org-b: {len(org_b)} rows, {org_b['is_ssrf'].sum()} SSRF")
    print(f"org-c: {len(org_c)} rows, {org_c['is_ssrf'].sum()} SSRF")