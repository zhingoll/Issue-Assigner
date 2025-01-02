import os
import sys
import argparse
import logging
import time
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import yaml
import pymongo
from pymongo.errors import PyMongoError, BulkWriteError
from github import Github, BadCredentialsException, RateLimitExceededException, UnknownObjectException
from dateutil.parser import parse as parse_date
from repofetcher import RepoFetcher

from github_models import (
    Repo,
    RepoIssue,
    ResolvedIssue,
    OpenIssue,
    ClosedPr,
    OpenPr,
    GitHubFetchLog,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load configuration information
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MONGO_URI = CONFIG["mongodb"]["uri"]
MONGO_DBNAME = CONFIG["mongodb"]["db"]
TOKENS = CONFIG["tokens"]


# Initialize database
client = pymongo.MongoClient(MONGO_URI)
db = client[MONGO_DBNAME]

repo_col = db["repos"]           # Stores repository information
repoissue_col = db["repo_issues"]  # Stores generic issue/PR information (is_pull=True or False)
resolvedissue_col = db["resolved_issues"]
openissue_col = db["open_issues"]
closedpr_col = db["closed_prs"]
openpr_col = db["open_prs"]
fetchlog_col = db["fetch_logs"]

# Check token validity
def check_tokens(tokens: List[str]) -> set:
    """
    Check if tokens are valid; invalid tokens are added to the invalid_tokens set.
    """
    invalid_tokens = set()
    for token in tokens:
        gh = Github(token, per_page=1)
        try:
            # Call get_user() once to test token validity
            _user = gh.get_user().login
            logger.debug("Token %s... is valid, user=%s", token[:6], _user)
        except BadCredentialsException:
            logger.warning("Token %s... is invalid (BadCredentials).", token[:6])
            invalid_tokens.add(token)
        except Exception as ex:
            logger.warning("Error checking token %s...: %s", token[:6], ex)
            invalid_tokens.add(token)
    return invalid_tokens

# Write failed number
def write_failed_number(number: int, item_type: str = "issue"):
    """
    Write the number of failed fetches for an issue or PR to a text file.
    """
    filename = "failed_issues.txt" if item_type == "issue" else "failed_prs.txt"
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"{item_type.capitalize()} Number: {number}\n")
    except Exception as e:
        logger.error("Failed to write to %s: %s", filename, e)


def insert_fetch_log(owner: str, name: str, user_github_login: Optional[str]) -> GitHubFetchLog:
    """
    Create and insert a log entry into fetch_logs.
    """
    doc = GitHubFetchLog(
        owner=owner,
        name=name,
        pid=os.getpid(),
        update_begin=datetime.now(timezone.utc),
        update_end=None,
        user_github_login=user_github_login
    )
    data = doc.dict()
    res = fetchlog_col.insert_one(data)
    data["_id"] = res.inserted_id
    return doc

def update_fetch_log(owner: str, name: str, updates: Dict[str, Any]):
    """
    Locate a log record based on (owner, name, update_end=None) and update it.
    """
    fetchlog_col.update_one(
        {"owner": owner, "name": name, "update_end": None},
        {"$set": updates}
    )


# Various Update Functions
def _update_repo_info(fetcher: RepoFetcher) -> Repo:
    """
    Fetch and update repository information (Repo).
    """
    logger.info("Updating repo info for %s/%s", fetcher.owner, fetcher.name)
    stats = fetcher.get_stats()

    if "languages" in stats and isinstance(stats["languages"], dict):
        lang_list = []
        for lang, cnt in stats["languages"].items():
            lang_list.append({"language": lang, "count": cnt})
        stats["languages"] = lang_list
    
    f_ = {"owner": stats["owner"], "name": stats["name"]}
    u_ = {
        "$setOnInsert": {
            "created_at": stats.get("repo_created_at")
        },
        "$set": {
            "language": stats.get("language"),
            "languages": stats.get("languages"),
            "description": stats.get("description"),
            "topics": stats.get("topics"),
            "readme": stats.get("readme"),
            "updated_at":None,
        }
    }
    repo_col.update_one(f_, u_, upsert=True)
    logger.info("Repo stats updated, rate=%s", fetcher.rate)

    return Repo(**stats)

def _update_issues(fetcher: RepoFetcher, since: datetime) -> List[Dict[str, Any]]:
    """
    Fetch all issues (including PRs, state=all), and upsert into repo_issues.
    """
    logger.info("Fetching issues for %s/%s since %s", fetcher.owner, fetcher.name, since)
    issues = fetcher.get_issues(since)
    logger.info("Fetched %d issues, rate=%s", len(issues), fetcher.rate)

    ops = []
    for iss in issues:
        f_ = {"owner": iss["owner"], "name": iss["name"], "number": iss["number"]}
        u_ = {"$set": iss}
        ops.append(pymongo.UpdateOne(f_, u_, upsert=True))

    if ops:
        try:
            repoissue_col.bulk_write(ops, ordered=False)
        except (PyMongoError, BulkWriteError) as e:
            logger.error("Bulk write for issues failed: %s", e)

    return issues

def _update_closed_issues(fetcher: RepoFetcher, nums: List[int], since: datetime) -> List[Dict[str, Any]]:
    """
    Update closed issues (not PRs), and write to resolved_issues.
    """
    logger.info("Updating closed issues for %s/%s, count=%d", fetcher.owner, fetcher.name, len(nums))
    ret = []

    query = {
        "owner": fetcher.owner,
        "name": fetcher.name,
        "is_pull": False,
        "state": "closed",
        "number": {"$in": nums}
    }
    closed_list = list(repoissue_col.find(query))
    logger.info("Found %d closed issues in DB", len(closed_list))

    for issue in closed_list:
        # If resolved_issues already has this record, skip
        if resolvedissue_col.find_one({"owner": issue["owner"], "name": issue["name"], "number": issue["number"]}):
            continue
        try:
            detail = fetcher.get_issue_detail(issue["number"])  # Includes events, resolver
        except ValueError as ve:
            logger.error("Failed to fetch closed issue #%d: %s", issue["number"], ve)
            write_failed_number(issue["number"], "issue")
            continue

        doc = {
            "owner": issue["owner"],
            "name": issue["name"],
            "number": issue["number"],
            "created_at": issue["created_at"],
            "resolved_at": issue.get("closed_at"),
            "resolver": detail["resolver"],       # list of potential resolvers
            "events": detail["events"],          # timeline events
            "issue_opener": issue["user"],
        }
        try:
            resolvedissue_col.insert_one(doc)
            ret.append(doc)
        except PyMongoError as e:
            logger.error("Insert resolved issue #%d failed: %s", issue["number"], e)

    return ret

def _update_open_issues(fetcher: RepoFetcher, nums: List[int], since: datetime) -> List[Dict[str, Any]]:
    """
    Update still open issues (not PRs), and write to open_issues.
    """
    logger.info("Updating open issues for %s/%s, count=%d", fetcher.owner, fetcher.name, len(nums))
    ret = []

    query = {
        "owner": fetcher.owner,
        "name": fetcher.name,
        "is_pull": False,
        "state": "open",
        "number": {"$in": nums}
    }
    open_list = list(repoissue_col.find(query))

    for issue in open_list:
        doc_in_db = openissue_col.find_one({
            "owner": issue["owner"],
            "name": issue["name"],
            "number": issue["number"]
        })
        if doc_in_db:
            openissue_col.update_one({"_id": doc_in_db["_id"]}, {"$set": {"updated_at": datetime.now(timezone.utc)}})
            ret.append(doc_in_db)
        else:
            try:
                detail = fetcher.get_issue_detail(issue["number"])
            except ValueError as ve:
                logger.error("Failed to fetch open issue #%d: %s", issue["number"], ve)
                write_failed_number(issue["number"], "issue")
                continue

            doc = {
                "owner": issue["owner"],
                "name": issue["name"],
                "number": issue["number"],
                "created_at": issue["created_at"],
                "updated_at": datetime.now(timezone.utc),
                "events": detail["events"],
                "issue_opener": issue["user"]
            }
            try:
                openissue_col.insert_one(doc)
                ret.append(doc)
            except PyMongoError as e:
                logger.error("Insert open issue #%d failed: %s", issue["number"], e)

    # Delete already closed issues
    # Query out state=closed issue.numbers, delete from open_issues
    closed_nums = repoissue_col.find({
        "owner": fetcher.owner,
        "name": fetcher.name,
        "is_pull": False,
        "state": "closed"
    }, {"number": 1})
    closed_list = [doc["number"] for doc in closed_nums]
    openissue_col.delete_many({
        "owner": fetcher.owner,
        "name": fetcher.name,
        "number": {"$in": closed_list}
    })

    return ret

def _update_closed_prs(fetcher: RepoFetcher, nums: List[int], since: datetime) -> List[Dict[str, Any]]:
    """
    Update closed PRs, and write to closed_prs.
    """
    logger.info("Updating closed PRs for %s/%s, count=%d", fetcher.owner, fetcher.name, len(nums))
    ret = []

    query = {
        "owner": fetcher.owner,
        "name": fetcher.name,
        "is_pull": True,
        "state": "closed",
        "number": {"$in": nums}
    }
    closed_pr_list = list(repoissue_col.find(query))
    for pr in closed_pr_list:
        if closedpr_col.find_one({"owner": pr["owner"], "name": pr["name"], "number": pr["number"]}):
            continue

        pr_detail = None
        attempts = 3
        while attempts > 0:
            try:
                pr_detail = fetcher.get_pr_detail(pr["number"])
                break
            except ValueError as ve:
                logger.error("Failed to fetch closed PR #%d: %s", pr["number"], ve)
                write_failed_number(pr["number"], "pr")
                break
            except Exception as ex:
                logger.error("Error fetching closed PR #%d: %s", pr["number"], ex)
                time.sleep(3)
            attempts -= 1

        if not pr_detail:
            continue

        doc = {
            "owner": pr["owner"],
            "name": pr["name"],
            "number": pr["number"],
            "created_at": pr["created_at"],
            "closed_at": pr.get("closed_at"),
            "reviewer_events": pr_detail["reviewer_events"],
            "normal_commenter_events": pr_detail["normal_commenter_events"],
            "label_events": pr_detail["label_events"],
            "pr_opener": pr["user"]
        }
        try:
            closedpr_col.insert_one(doc)
            ret.append(doc)
        except PyMongoError as e:
            logger.error("Insert closed PR #%d failed: %s", pr["number"], e)

    return ret

def _update_open_prs(fetcher: RepoFetcher, nums: List[int], since: datetime) -> List[Dict[str, Any]]:
    """
    Update still open PRs, and write to open_prs.
    """
    logger.info("Updating open PRs for %s/%s, count=%d", fetcher.owner, fetcher.name, len(nums))
    ret = []

    query = {
        "owner": fetcher.owner,
        "name": fetcher.name,
        "is_pull": True,
        "state": "open",
        "number": {"$in": nums}
    }
    open_pr_list = list(repoissue_col.find(query))
    for pr in open_pr_list:
        exists = openpr_col.find_one({"owner": pr["owner"], "name": pr["name"], "number": pr["number"]})
        if exists:
            openpr_col.update_one(
                {"_id": exists["_id"]},
                {"$set": {"updated_at": datetime.now(timezone.utc)}}
            )
            ret.append(exists)
        else:
            pr_detail = None
            attempts = 3
            while attempts > 0:
                try:
                    pr_detail = fetcher.get_pr_detail(pr["number"])
                    break
                except ValueError as ve:
                    logger.error("Failed to fetch open PR #%d: %s", pr["number"], ve)
                    write_failed_number(pr["number"], "pr")
                    break
                except Exception as ex:
                    logger.error("Error fetching open PR #%d: %s", pr["number"], ex)
                    time.sleep(3)
                attempts -= 1

            if not pr_detail:
                continue

            doc = {
                "owner": pr["owner"],
                "name": pr["name"],
                "number": pr["number"],
                "created_at": pr["created_at"],
                "updated_at": datetime.now(timezone.utc),
                "reviewer_events": pr_detail["reviewer_events"],
                "normal_commenter_events": pr_detail["normal_commenter_events"],
                "label_events": pr_detail["label_events"],
                "pr_opener": pr["user"]
            }
            try:
                openpr_col.insert_one(doc)
                ret.append(doc)
            except PyMongoError as e:
                logger.error("Insert open PR #%d failed: %s", pr["number"], e)

    # Clean up already closed PRs
    closed_nums_cursor = repoissue_col.find({
        "owner": fetcher.owner,
        "name": fetcher.name,
        "is_pull": True,
        "state": "closed"
    }, {"number": 1})
    closed_list = [item["number"] for item in closed_nums_cursor]
    openpr_col.delete_many({
        "owner": fetcher.owner,
        "name": fetcher.name,
        "number": {"$in": closed_list}
    })

    return ret

# Core logic for updating a repository
def update_repo(tokens: List[str], owner: str, name: str,user_github_login: Optional[str] = None):
    """
    Update all information for a repository.
    Includes:
      - Updating Repo
      - Updating Issues
      - Updating Closed Issues
      - Updating Open Issues
      - Updating Closed PRs
      - Updating Open PRs
      - Finally updating fetch_logs
    """

    # Insert fetch_log
    log_entry = insert_fetch_log(owner, name, user_github_login)
    logger.info("Start updating %s/%s with tokens: %s...", owner, name, [t[:6]+"..." for t in tokens])

    fetcher = RepoFetcher(tokens, owner, name)

    # 1) Update repo
    repo_stats = _update_repo_info(fetcher)

    # 2) Determine since
    repo_db = repo_col.find_one({"owner": owner, "name": name})
    if repo_db["updated_at"] is None:
        since_dt = repo_stats.repo_created_at if hasattr(repo_stats, "repo_created_at") else datetime(1970,1,1, tzinfo=timezone.utc) 
    else:
        since_dt = repo_db["updated_at"]

    # if isinstance(since_dt, str):
    #     since_dt = parse_date(since_dt)
    #     if since_dt.tzinfo is None or since_dt.tzinfo.utcoffset(since_dt) is None:
    #         since_dt = since_dt.replace(tzinfo=timezone.utc)

    # 3) Fetch + update issues (including PRs)
    all_issues = _update_issues(fetcher, since_dt)
    update_fetch_log(owner, name, {
        "updated_issues": len(all_issues),
        "rate": fetcher.rate[2],
        "rate_repo_stat": fetcher.rate[2],
    })

    # 4) Distinguish based on state / is_pull:
    closed_issue_nums = [i["number"] for i in all_issues if i["state"] == "closed" and not i["is_pull"]]
    open_issue_nums = [i["number"] for i in all_issues if i["state"] == "open" and not i["is_pull"]]
    closed_pr_nums = [i["number"] for i in all_issues if i["state"] == "closed" and i["is_pull"]]
    open_pr_nums = [i["number"] for i in all_issues if i["state"] == "open" and i["is_pull"]]

    # a. Update closed issues
    closed_issues = _update_closed_issues(fetcher, closed_issue_nums, since_dt)
    update_fetch_log(owner, name, {
        "updated_resolved_issues": len(closed_issues),
        "rate": fetcher.rate[2]
    })

    # b. Update open issues
    open_issues = _update_open_issues(fetcher, open_issue_nums, since_dt)
    update_fetch_log(owner, name, {
        "updated_open_issues": len(open_issues),
        "rate": fetcher.rate[2]
    })

    # c. Update closed PRs
    closed_prs = _update_closed_prs(fetcher, closed_pr_nums, since_dt)
    update_fetch_log(owner, name, {
        "updated_closed_prs": len(closed_prs),
        "rate": fetcher.rate[2]
    })

    # d. Update open PRs
    open_prs = _update_open_prs(fetcher, open_pr_nums, since_dt)
    update_fetch_log(owner, name, {
        "updated_open_prs": len(open_prs),
        "rate": fetcher.rate[2]
    })

    # Update the 'updated_at' field of the repository to the current time
    repo_col.update_one(
        {"owner": owner, "name": name},
        {"$set": {"updated_at": datetime.now(timezone.utc)}}
    )

    # Finally conclude the log
    fetchlog_col.update_one(
        {"owner": owner, "name": name, "update_end": None},
        {"$set": {"update_end": datetime.now(timezone.utc)}}
    )
    logger.info("Finished updating %s/%s.", owner, name)

# update repo under tokens
def update_under_tokens(tokens: List[str], repos: List[str]) -> None:
    """
    Update multiple repositories with given tokens in a single process.
    """
    logger.info("Process %s handling repos: %s", os.getpid(), repos)
    logger.info("Using update_time: %s")
    for r in repos:
        owner, name = r.split("/")
        update_repo(tokens, owner, name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug info")
    parser.add_argument("--nprocess", type=int, default=mp.cpu_count(), help="Number of processes")
    parser.add_argument("--repos", type=str, default="X-lab2017/open-digger", help="Comma-separated repos to update")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not args.repos:
        repos = ["X-lab2017/opendigger"]
    else:
        repos = args.repos.split(",")

    # Validate tokens
    invalid_tokens = check_tokens(TOKENS)
    valid_tokens = list(set(TOKENS) - invalid_tokens)
    if not valid_tokens:
        logger.error("No valid tokens found, exit.")
        sys.exit(1)

    logger.info("Valid tokens: %s", [t[:6]+"..." for t in valid_tokens])
    logger.info("Start data update at %s", datetime.now())

    # Allocate tokens to repositories (here, every two tokens are allocated to several repositories as an example)
    params = defaultdict(list)
    for i, project in enumerate(repos):
        t1 = valid_tokens[i % len(valid_tokens)]
        t2 = valid_tokens[(i + 1) % len(valid_tokens)]
        params[(t1, t2)].append(project)

    # Concurrent multi-process
    pool_size = min(args.nprocess, len(params))
    logger.info("Create pool of size %d for parallel updates", pool_size)
    with mp.Pool(pool_size) as pool:
        pool.starmap(update_under_tokens, params.items())

    logger.info("All tasks finished at %s", datetime.now())

if __name__ == "__main__":
    main()



