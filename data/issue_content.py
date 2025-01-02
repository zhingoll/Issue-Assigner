import os
import re
import logging
import argparse
import yaml
import pymongo
import multiprocessing as mp
from typing import List, Optional
from datetime import datetime
from dateutil.parser import parse as parse_date

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MONGO_URI = CONFIG["mongodb"]["uri"]
MONGO_DBNAME = CONFIG["mongodb"]["db"]

client = pymongo.MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]

repo_issues = db["repo_issues"]            
issue_contents = db["issue_contents"]      
repos_col = db["repos"]                    # If you need to iterate over repos


def _delete_code_snippets(s: Optional[str]) -> str:
    """Remove code blocks (```...```) from text."""
    if s is None:
        return ""
    pattern = re.compile(r"```.*?```", flags=re.S)
    return pattern.sub("", s)


def _delete_urls(s: Optional[str]) -> str:
    """Remove URLs (http/https) from text."""
    if s is None:
        return ""
    pattern = re.compile(r"https?://[\w\.-]+(?:\:\d+)?(?:/[\w\./?%&=\-]*)?")
    return pattern.sub("", s)


def update_issue_content(owner: str, name: str, issue_number: int) -> None:
    """
    Insert or skip a single issue into the `issue_contents` collection.
    - Skips if it's already in `issue_contents`.
    - Skips if the doc is actually a pull request (`is_pull=True`).
    """
    # Check if already exists in `issue_contents`
    existing = issue_contents.find_one({
        "owner": owner,
        "name": name,
        "number": issue_number
    })
    if existing:
        logger.info(f"{owner}/{name}#{issue_number} already in issue_contents, skip.")
        return

    # Fetch from `repo_issues`
    repo_issue = repo_issues.find_one({
        "owner": owner,
        "name": name,
        "number": issue_number
    })

    if not repo_issue:
        logger.warning(f"{owner}/{name}#{issue_number} not found in repo_issues.")
        return

    # If it's a pull request, skip
    if repo_issue.get("is_pull") is True:
        logger.info(f"{owner}/{name}#{issue_number} is a PR, skipping.")
        return

    # Clean body
    raw_body = repo_issue.get("body", "")
    clean_body = _delete_urls(_delete_code_snippets(raw_body))

    # Prepare new doc for `issue_contents`
    doc = {
        "owner": owner,
        "name": name,
        "number": issue_number,
        "title": repo_issue.get("title", ""),
        "body": clean_body
    }
    try:
        issue_contents.insert_one(doc)
        logger.info(f"{owner}/{name}#{issue_number} -> inserted into issue_contents.")
    except Exception as exc:
        logger.error(f"Failed inserting {owner}/{name}#{issue_number} into issue_contents: {exc}")


def update_dataset_with_issues(owner: str, name: str) -> None:
    """
    For all issues in `repo_issues` for a given repo (owner, name),
    call update_issue_content.
    """
    query = {"owner": owner, "name": name}
    all_issues_cursor = repo_issues.find(query, {"owner":1, "name":1, "number":1, "is_pull":1})
    all_issues = list(all_issues_cursor)

    logger.info(f"Found {len(all_issues)} issues/PRs in repo_issues for {owner}/{name}.")

    for i, issue in enumerate(all_issues, start=1):
        update_issue_content(owner, name, issue["number"])
        if i % 50 == 0:
            logger.info(f"Processed {i} issues for {owner}/{name}.")


def get_dataset_for_repo(
    owner: str,
    name: str,
):
    """
    Update the dataset for a single repo, i.e. re-check its issues and store
    them into `issue_contents`.
    """
    logger.info(f"Start dataset update for {owner}/{name}")

    update_dataset_with_issues(owner, name)

    logger.info(f"Finished dataset update for {owner}/{name}")

# Load configuration file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_dataset_all(n_process: Optional[int] = None):
    """
    Update the dataset for all repos or for a hard-coded list if you prefer.
    If n_process is provided, do it in parallel.
    """
    # Example: If you want to loop over all repos in your 'repos' collection, do:
    # all_repos = list(repos_col.find({}))
    # or if you only want a single repo for demonstration:

    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(CONFIG_FILE)
    all_repos = [(config['repo']['owner'], config['repo']['name'])]


    if not n_process:
        # Single-process
        for (owner, name) in all_repos:
            get_dataset_for_repo(owner, name)
        logger.info("Completed all repos in single-process mode.")
    else:
        # Multi-process approach
        logger.info(f"Starting multi-process with pool size = {n_process}")
        # Prepare parameters for each repo
        params = [(owner, name) for (owner, name) in all_repos]

        with mp.Pool(n_process) as pool:
            pool.starmap(get_dataset_for_repo, params)

        logger.info("Completed all repos in multi-process mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocess", type=int, default=None,
                        help="Number of processes to use for parallel processing.")
    args = parser.parse_args()
    n_process = args.nprocess

    logger.info("Start dataset-building script!")
    get_dataset_all(n_process)
    logger.info("Finish dataset-building script!")
