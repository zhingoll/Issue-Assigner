import pandas as pd
import yaml
from pymongo import MongoClient
from neo4j import GraphDatabase
import os

# Load configuration file
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Connect to Neo4j
def get_neo4j_driver(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))

# Connect to MongoDB
def get_mongo_client(uri):
    return MongoClient(uri)

# Fetch data from Neo4j
def fetch_data_from_neo4j(driver, query):
    with driver.session() as session:
        result = session.run(query)
        data = pd.DataFrame([r.values() for r in result], columns=result.keys())
    return data

# Load Issue content data from MongoDB
def load_issue_content(owner, name, issues_collection):
    query = {"owner": owner, "name": name}
    issue_contents = list(issues_collection.find(query))
    issue_contents_df = pd.DataFrame(issue_contents)
    issue_contents_df = issue_contents_df[['number', 'title', 'body']]
    return issue_contents_df

# Load resolved Issue data from MongoDB
def load_resolved_issues(owner, name, resolved_issues_collection):
    query = {"owner": owner, "name": name}
    resolved_issues = list(resolved_issues_collection.find(query))
    resolved_issues_df = pd.DataFrame(resolved_issues)
    resolved_issues_df = resolved_issues_df[['number', 'resolver', 'resolved_at']]
    return resolved_issues_df

# Load open Issue data from MongoDB
def load_open_issues(owner, name, repo_issues_collection):
    query = {"owner": owner, "name": name, "state": "open"}
    open_issues = list(repo_issues_collection.find(query))
    open_issues_df = pd.DataFrame(open_issues)
    open_issues_df = open_issues_df[['user','number', 'title', 'body']]
    return open_issues_df

def main():
    # Load the configuration file
    CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(CONFIG_FILE)

    # Initialize Neo4j driver and MongoDB client
    neo4j_driver = get_neo4j_driver(config['neo4j']['uri'], config['neo4j']['username'], config['neo4j']['password'])
    mongo_client = get_mongo_client(config['mongodb']['url'])
    
    # Get collections from MongoDB
    db = mongo_client[config['mongodb']['db']]
    issues_collection = db['issue_contents']
    resolved_issues_collection = db['resolved_issues']
    repo_issues_collection = db['repo_issues']
    
    # Conver 'user-pr-issue' to 'user-issue'
    query = f"""
    // Query user interactions directly related to Issues
    MATCH (u:User)-[r]->(i)
    WHERE (i:OpenIssue OR i:ResolvedIssue) AND NOT u.name ENDS WITH '[bot]' 
    AND i.repo = '{config['repo']['name']}' AND NOT type(r) = 'UNLABELED'
    RETURN u.name AS UserName, 
           CASE WHEN type(r) = 'OPENED' THEN 'ISSUE_OPEN' ELSE type(r) END AS EventType, 
           i.number AS IssueNumber, 
           i.created_at AS IssueCreatedTime

    UNION

    // Query user activities via PR cross-referencing to Issues
    MATCH (u:User)-[r]->(pr)<-[:`CROSS-REFERENCED`]-(i)
    WHERE (pr:OpenPR OR pr:ClosedPR) AND (i:OpenIssue OR i:ResolvedIssue) AND NOT u.name ENDS WITH '[bot]' 
    AND pr.repo = '{config['repo']['name']}' AND NOT type(r) = 'UNLABELED'
    RETURN u.name AS UserName, 
           CASE WHEN type(r) = 'OPENED' THEN 'PR_OPEN' ELSE type(r) END AS EventType, 
           i.number AS IssueNumber, 
           i.created_at AS IssueCreatedTime
    """
    
    # Retrieve data from Neo4j
    data = fetch_data_from_neo4j(neo4j_driver, query)
    data.to_csv('user_issue.csv', index=False)

    # Load Issue data from MongoDB
    open_issues_df = load_open_issues(config['repo']['owner'], config['repo']['name'], repo_issues_collection)
    open_issues_df.to_csv('open_issues.csv', index=False)

    resolved_issues_df = load_resolved_issues(config['repo']['owner'], config['repo']['name'], resolved_issues_collection)
    resolved_issues_df.to_csv('resolved_issues.csv', index=False)

    issue_content_df = load_issue_content(config['repo']['owner'], config['repo']['name'], issues_collection)
    issue_content_df.to_csv('issue_content.csv', index=False)

if __name__ == "__main__":
    main()
