from urllib.parse import urlparse


def parse_owner_repo(repo_url: str) -> tuple[str, str]:
    path = urlparse(repo_url).path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub repo url: {repo_url}")
    return parts[-2], parts[-1]
