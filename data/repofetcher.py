import os
import time
import logging
import yaml
import pymongo

from typing import Callable, TypeVar, Optional, Any, Dict, List, Tuple
from datetime import datetime, timezone
from calendar import monthrange
from dateutil.parser import parse as parse_date
from github import Github
from github import RateLimitExceededException, UnknownObjectException, BadCredentialsException
from github.GithubObject import NotSet
from collections import defaultdict

logger = logging.getLogger(__name__)
T = TypeVar("T")


# Read the configuration and initialize the database
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MONGO_URI = CONFIG["mongodb"]["uri"]
MONGO_DBNAME = CONFIG["mongodb"]["db"]

mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client[MONGO_DBNAME]
repoissue_col = db["repo_issues"]

def get_page_num(per_page: int, total_count: int) -> int:
    """
    Calculate the total number of pages required based on per_page and total_count.
    """
    if per_page <= 0:
        return 0
    return (total_count + per_page - 1) // per_page

def get_month_interval(date: datetime) -> Tuple[datetime, datetime]:
    """
    Given a datetime, return the start and end time interval [month_start, month_end] of the month.
    """
    if date.tzinfo is None:
        logger.warning("date is not timezone aware: %s", date)
        date = date.replace(tzinfo=timezone.utc)
    date = date.astimezone(timezone.utc)
    since = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    until = date.replace(
        day=monthrange(date.year, date.month)[1],
        hour=23,
        minute=59,
        second=59,
        microsecond=999999,
    )
    return since, until


class RepoFetcher:
    """
    A class for fetching repository data from GitHub, supporting multiple token rotation, retries, 
    and querying the database through pymongo.
    """

    def __init__(self, tokens: List[str], owner: str, name: str):
        self.tokens = tokens
        self.current_token_index = 0
        self.gh = self.create_github_client()
        self.gh.per_page = 100

        self.owner = owner
        self.name = name

        # Retrieve the repo
        self.repo = self.request_github(lambda: self.gh.get_repo(f"{owner}/{name}"))
        if not self.repo:
            raise ValueError(f"Failed to get repo {owner}/{name}")

        self.owner = self.repo.owner.login
        self.name = self.repo.name

        # Retrieve initial rate information
        rate_info = self.request_github(lambda: self.gh.rate_limiting, default=(0, 0))
        self.rate_remaining, self.rate_limit = rate_info
        self.rate_consumed = 0

        # Record the repository creation time
        self.created_at = self.repo.created_at.astimezone(timezone.utc)

    def create_github_client(self) -> Github:
        """Create a GitHub client based on the current token"""
        return Github(self.tokens[self.current_token_index], per_page=100)

    def request_github(self, gh_func: Callable[..., T], params: Tuple = (), default: Any = None) -> Optional[T]:
        retries = 3
        while retries > 0:
            try:
                return gh_func(*params)
            except RateLimitExceededException as ex:
                logger.warning("Rate limit exceeded, rotating token: %s", ex)
                time.sleep(1)
                self.rotate_token()
            except UnknownObjectException as ex:
                logger.error("Unknown object exception: %s", ex)
                return None
            except BadCredentialsException as ex:
                logger.warning("Bad credentials for token %s..., rotating. %s", self.tokens[self.current_token_index][:6], ex)
                self.rotate_token()
            except Exception as ex:
                logger.error("Exception in request_github: %s", ex)
                logger.error("gh_func=%s, params=%s", gh_func, params)
                time.sleep(3)
            retries -= 1
        return default


    def rotate_token(self):
        """Switch to the next token and re-fetch self.repo"""
        old_idx = self.current_token_index
        self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
        logger.info("Switch from token index %d to %d: %s...", old_idx, self.current_token_index, self.tokens[self.current_token_index][:6])

        self.gh = self.create_github_client()
        self.gh.per_page = 100
        self.repo = self.request_github(lambda: self.gh.get_repo(f"{self.owner}/{self.name}"))
        if self.repo:
            self.owner = self.repo.owner.login
            self.name = self.repo.name
            self.created_at = self.repo.created_at.astimezone(timezone.utc)

    def _update_rate_stats(self):
        """Update remaining and consumed rate counts"""
        old_remaining = self.rate_remaining
        info = self.request_github(lambda: self.gh.rate_limiting, default=(0, 0))
        new_remaining, new_limit = info
        if new_limit > 0:
            # If the remaining count increases, it means the token or quota has been reset
            if old_remaining >= new_remaining:
                self.rate_consumed += (old_remaining - new_remaining)
            else:
                self.rate_consumed += old_remaining + new_limit - new_remaining
            self.rate_remaining = new_remaining
            self.rate_limit = new_limit

    @property
    def rate(self) -> Tuple[int, int, int]:
        """Return (rate_remaining, rate_limit, rate_consumed)"""
        return (self.rate_remaining, self.rate_limit, self.rate_consumed)

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve basic repository information, including languages, readme, etc.
        """
        result = self.request_github(
            lambda: {
                "owner": self.repo.owner.login,
                "name": self.repo.name,
                "language": self.repo.language,
                "languages": self.repo.get_languages(),
                "repo_created_at": self.created_at,
                "description": self.repo.description,
                "topics": self.repo.get_topics(),
                "readme": self.repo.get_readme().decoded_content.decode("utf-8", "ignore"),
                "updated_at": datetime.now(timezone.utc),
            },
            default={}
        )
        self._update_rate_stats()
        return result or {}

    def get_stars(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve star events after the specified 'since' date.
        """
        results = []
        stargazers = self.request_github(lambda: self.repo.get_stargazers_with_dates(), default=[])
        if not stargazers:
            return results

        total_count = getattr(stargazers, "totalCount", 0)
        page_num = get_page_num(self.gh.per_page, total_count)

        for p in range(page_num):
            logger.debug("star page %d / %d, rate=%s", p, page_num, self.gh.rate_limiting)
            page_data = self.request_github(stargazers.get_page, (p,), [])
            for star in page_data:
                starred_at = star.starred_at.astimezone(timezone.utc)
                if starred_at < since:
                    self._update_rate_stats()
                    return results
                user_login = star.user.login if star.user else None
                results.append({
                    "owner": self.owner,
                    "name": self.name,
                    "user": user_login,
                    "starred_at": starred_at,
                })

        self._update_rate_stats()
        return results

    def get_commits_in_month(self, date: datetime) -> int:
        """
        Return the total number of commits for a specified month.
        """
        since, until = get_month_interval(date)
        total = self.request_github(
            lambda: self.repo.get_commits(since=since, until=until).totalCount,
            default=0
        )
        self._update_rate_stats()
        return total

    def get_commits(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve all commits after the specified 'since' time.
        """
        results = []
        commits = self.request_github(lambda: self.repo.get_commits(since=since), default=[])
        if not commits:
            return results

        total_count = getattr(commits, "totalCount", 0)
        page_num = get_page_num(self.gh.per_page, total_count)

        for p in range(page_num):
            logger.debug("commit page %d / %d, rate=%s", p, page_num, self.gh.rate_limiting)
            page_data = self.request_github(commits.get_page, (p,), [])
            for c in page_data:
                try:
                    author = c.author.login if c.author else None
                except:
                    author = None
                try:
                    committer = c.committer.login if c.committer else None
                except:
                    committer = None
                results.append({
                    "owner": self.owner,
                    "name": self.name,
                    "sha": c.sha,
                    "author": author,
                    "authored_at": c.commit.author.date.astimezone(timezone.utc),
                    "committer": committer,
                    "committed_at": c.commit.committer.date.astimezone(timezone.utc),
                    "message": c.commit.message,
                })

        self._update_rate_stats()
        return results

    def get_issue(self, number: int) -> List[Dict[str, Any]]:
        """
        Retrieve core fields of a single issue (or PR).
        """
        results = []
        issue_obj = self.request_github(lambda: self.repo.get_issue(number), default=None)
        if not issue_obj:
            return results

        closed_at = None
        if issue_obj.state == "closed" and issue_obj.closed_at:
            closed_at = issue_obj.closed_at.astimezone(timezone.utc)

        is_pull = (issue_obj._pull_request != NotSet)
        merged_at = None
        if is_pull and issue_obj.pull_request and issue_obj.pull_request.raw_data.get("merged_at"):
            merged_str = issue_obj.pull_request.raw_data["merged_at"]
            if merged_str:
                merged_at = parse_date(merged_str).astimezone(timezone.utc)

        user_login = issue_obj.user.login if issue_obj.user else None
        results.append({
            "owner": self.owner,
            "name": self.name,
            "number": issue_obj.number,
            "user": user_login,
            "state": issue_obj.state,
            "created_at": issue_obj.created_at.astimezone(timezone.utc),
            "closed_at": closed_at,
            "title": issue_obj.title,
            "body": issue_obj.body,
            "labels": [lab.name for lab in issue_obj.labels],
            "is_pull": is_pull,
            "merged_at": merged_at,
        })

        self._update_rate_stats()
        return results

    def get_issues(self, since: datetime) -> List[Dict[str, Any]]:
        """
        Retrieve all issues (including PRs, state=all) starting from the specified 'since' time.
        """
        results = []
        issues = self.request_github(
            lambda: self.repo.get_issues(since=since, direction="asc", state="all"),
            default=[]
        )
        if not issues:
            print("not issues!!!")
            return results

        total_count = getattr(issues, "totalCount", 0)
        page_cnt = get_page_num(self.gh.per_page, total_count)

        for p in range(page_cnt):
            logger.debug("issue page %d / %d, rate=%s", p, page_cnt, self.gh.rate_limiting)
            page_data = self.request_github(issues.get_page, (p,), [])
            for issue_obj in page_data:
                closed_at = None
                if issue_obj.state == "closed" and issue_obj.closed_at:
                    closed_at = issue_obj.closed_at.astimezone(timezone.utc)
                is_pull = (issue_obj._pull_request != NotSet)
                merged_at = None
                if is_pull and issue_obj.pull_request and issue_obj.pull_request.raw_data.get("merged_at"):
                    merged_str = issue_obj.pull_request.raw_data["merged_at"]
                    if merged_str:
                        merged_at = parse_date(merged_str).astimezone(timezone.utc)

                user_login = issue_obj.user.login if issue_obj.user else None
                results.append({
                    "owner": self.owner,
                    "name": self.name,
                    "number": issue_obj.number,
                    "user": user_login,
                    "state": issue_obj.state,
                    "created_at": issue_obj.created_at.astimezone(timezone.utc),
                    "closed_at": closed_at,
                    "title": issue_obj.title,
                    "body": issue_obj.body,
                    "labels": [lab.name for lab in issue_obj.labels],
                    "is_pull": is_pull,
                    "merged_at": merged_at,
                })

        self._update_rate_stats()
        return results

    def get_issue_with_retries(self, number: int):
        """
        Attempt multiple times to retrieve a single issue (or PR). Throw an exception if unsuccessful.
        """
        attempts = 3
        while attempts > 0:
            issue_obj = self.request_github(self.repo.get_issue, (number,), None)
            if issue_obj:
                return issue_obj
            attempts -= 1
            logger.info("Issue #%d not found, %d attempts left...", number, attempts)
            time.sleep(3)

        raise ValueError(f"Issue #{number} not found after multiple attempts")

    def get_issue_detail(self, number: int) -> Dict[str, Any]:
        """
        Retrieve detailed information of an Issue (timeline events + potential resolvers)
        Includes cross-referenced times, searching for closed PRs in repo_issues via pymongo, adding their user to resolver["pr"].
        """
        issue_obj = self.get_issue_with_retries(number)
        if not issue_obj:
            raise ValueError(f"Issue #{number} not found")

        events = []
        timeline = self.request_github(issue_obj.get_timeline, default=[])
        if not timeline:
            self._update_rate_stats()
            return {
                "owner": self.owner,
                "name": self.name,
                "number": number,
                "events": [],
                "resolver": []
            }

        total_count = getattr(timeline, "totalCount", 0)
        page_cnt = get_page_num(self.gh.per_page, total_count)

        resolver_map = {
            "assignee": [],
            "pr": [],
            "commenter": []
        }

        for p in range(page_cnt):
            page_data = self.request_github(timeline.get_page, (p,), [])
            for ev_obj in page_data:
                ev_raw = ev_obj.raw_data
                ev_type = ev_raw.get("event", "")
                additional = {}

                if ev_type in ["assigned", "unassigned"]:
                    assignee = ev_raw.get("assignee")
                    if assignee:
                        additional["assignee"] = assignee["login"]
                        resolver_map["assignee"].append(assignee["login"])
                elif ev_type in ["labeled", "unlabeled"]:
                    label_ = ev_raw.get("label", {})
                    additional["label"] = label_.get("name")
                elif ev_type == "cross-referenced":
                    # Check if the cross-referenced issue is a closed PR
                    src_issue_data = ev_raw.get("source", {}).get("issue", {})
                    cross_num = src_issue_data.get("number")
                    if cross_num:
                        additional["source"] = cross_num
                        # Search for the closed PR in repo_issues
                        found_pr = repoissue_col.find_one({
                            "owner": self.owner,
                            "name": self.name,
                            "is_pull": True,
                            "state": "closed",
                            "number": cross_num
                        })
                        if found_pr:
                            # If the corresponding closed PR is found, consider its user as a potential resolver
                            pr_user = found_pr.get("user")
                            if pr_user:
                                resolver_map["pr"].append(pr_user)
                elif ev_type == "commented":
                    user_ = ev_raw.get("user", {})
                    commenter = user_.get("login")
                    additional["comment"] = ev_raw.get("body")
                    additional["commenter"] = commenter
                    resolver_map["commenter"].append(commenter)
                elif ev_type == "referenced":
                    additional["commit"] = ev_raw.get("commit_id")

                t_str = ev_raw.get("created_at")
                t_ = parse_date(t_str).astimezone(timezone.utc) if t_str else None

                actor_data = ev_raw.get("actor")
                actor_login = actor_data.get("login") if actor_data else None

                events.append({
                    "type": ev_type,
                    "time": t_,
                    "actor": actor_login,
                    **additional
                })

        # Remove duplicates
        resolver_map["assignee"] = list(set(resolver_map["assignee"]))
        resolver_map["pr"] = list(set(resolver_map["pr"]))
        resolver_map["commenter"] = list(set(resolver_map["commenter"]))

        # Simple prioritization
        if resolver_map["assignee"]:
            final_resolver = resolver_map["assignee"]
        elif resolver_map["pr"]:
            final_resolver = resolver_map["pr"]
        elif resolver_map["commenter"]:
            final_resolver = resolver_map["commenter"]
        else:
            final_resolver = []

        self._update_rate_stats()
        return {
            "owner": self.owner,
            "name": self.name,
            "number": number,
            "events": events,
            "resolver": final_resolver
        }

    def get_pr_with_retries(self, number: int):
        """
        Attempt multiple times to retrieve a PR. Throw an exception if unsuccessful.
        """
        attempts = 3
        while attempts > 0:
            pr_obj = self.request_github(self.repo.get_pull, (number,), None)
            if pr_obj:
                return pr_obj
            attempts -= 1
            logger.info("PR #%d not found, %d attempts left...", number, attempts)
            time.sleep(3)
        raise ValueError(f"PR #{number} not found after multiple attempts")

    def get_pr_detail(self, number: int) -> Dict[str, Any]:
        """
        Retrieve detailed information of a PR, including reviewer_events, normal_commenter_events, label_events
        """
        pr_obj = self.get_pr_with_retries(number)
        if not pr_obj:
            raise ValueError(f"PR #{number} not found")

        reviews = self.request_github(pr_obj.get_reviews, default=[])
        review_comments = self.request_github(pr_obj.get_review_comments, default=[])
        comment_map = defaultdict(list)

        # Group review_comments by user.login
        for rc in review_comments:
            if rc.user:
                comment_map[rc.user.login].append({
                    "time": rc.created_at.astimezone(timezone.utc),
                    "comment": rc.body
                })

        reviewer_events = []
        for rev in reviews:
            if not rev.user:
                continue
            user_login = rev.user.login
            if user_login in comment_map:
                for cmt in comment_map[user_login]:
                    reviewer_events.append({
                        "type": "review_comment",
                        "time": cmt["time"],
                        "actor": user_login,
                        "comment": cmt["comment"]
                    })
            else:
                reviewer_events.append({
                    "type": "review_comment",
                    "time": None,
                    "actor": user_login,
                    "comment": None
                })

        # Normal comments
        normal_commenter_events = []
        normal_comments = self.request_github(pr_obj.get_issue_comments, default=[])
        for cmt in normal_comments:
            normal_commenter_events.append({
                "type": "normal_comment",
                "time": cmt.created_at.astimezone(timezone.utc),
                "actor": cmt.user.login if cmt.user else None,
                "comment": cmt.body
            })

        # label events
        label_events = []
        pr_events = self.request_github(pr_obj.get_issue_events, default=[])
        if pr_events:
            total_count = getattr(pr_events, "totalCount", 0)
            page_cnt = get_page_num(self.gh.per_page, total_count)
            for p in range(page_cnt):
                page_data = self.request_github(pr_events.get_page, (p,), [])
                for ev in page_data:
                    ev_raw = ev.raw_data
                    ev_type = ev_raw.get("event", "")
                    if ev_type in ["labeled", "unlabeled"]:
                        lbl = ev_raw.get("label", {})
                        lbl_name = lbl.get("name")
                        actor_ = ev_raw.get("actor", {})
                        actor_login = actor_.get("login")
                        ctime = ev_raw.get("created_at")
                        t_ = parse_date(ctime).astimezone(timezone.utc) if ctime else None

                        label_events.append({
                            "type": ev_type,
                            "time": t_,
                            "actor": actor_login,
                            "comment": lbl_name
                        })

        self._update_rate_stats()
        return {
            "owner": self.owner,
            "name": self.name,
            "number": number,
            "reviewer_events": reviewer_events,
            "normal_commenter_events": normal_commenter_events,
            "label_events": label_events,
        }
