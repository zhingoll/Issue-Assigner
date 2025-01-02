from pymongo import MongoClient
from py2neo import Graph, Node, Relationship
from datetime import datetime
import yaml
import os
import os

def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Load configuration from the configuration file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
config = load_config(CONFIG_FILE)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
config = load_config(CONFIG_FILE)

# MongoDB configuration
mongo_uri = config['mongodb']['uri']
mongo_db_name = config['mongodb']['db']
mongo_client = MongoClient(mongo_uri)
db = mongo_client[mongo_db_name]

# Neo4j configuration
neo4j_uri = config['neo4j']['uri']
neo4j_username = config['neo4j']['username']
neo4j_password = config['neo4j']['password']
graph = Graph(neo4j_uri, auth=(neo4j_username, neo4j_password))
# repo_issues = db['repo_issue']

# Use for deletion
# Delete nodes and relationships in batches
def delete_in_batches(graph, batch_size):
    query = """
    MATCH (n)
    WITH n LIMIT $batchSize
    DETACH DELETE n
    RETURN count(n) as deletedCount
    """
    return graph.run(query, batchSize=batch_size).evaluate()  # Execute the query and return the number of nodes deleted

# Loop until no data is left to delete
total_deleted = 0
while True:
    deleted_count = delete_in_batches(graph, 10000) 
    if deleted_count == 0:
        break
    total_deleted += deleted_count
    print(f"Deleted {deleted_count} nodes in this batch, total deleted: {total_deleted}")

print("All data deleted successfully.")

def format_datetime(dt):
    """Helper function to format datetime objects for Neo4j."""
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%dT%H:%M:%S')
    return dt  # If it's not a datetime object, return as is


# Add a user node
def add_user_node(username):
    try:
        user_node = Node("User", name=username)
        graph.merge(user_node, "User", "name")
    except Exception as e:
        print(f"An add_user_node error occurred: {e}")
    return user_node

# Add OpenIssue or ClosedIssue node
def add_open_closed_issue_node(doc, type):
    try:
        properties = {
            'number': doc['number'],
            'created_at': format_datetime(doc['created_at']),
            'repo': doc['name']
        }
        if type == 'OpenIssue':
            properties['updated_at'] = format_datetime(doc.get('updated_at', None))
        elif type == 'ResolvedIssue':
            properties['resolved_at'] = format_datetime(doc.get('resolved_at', None))
            properties['resolver'] = doc.get('resolver', None)

        node = Node(type, **properties)
        graph.merge(node, type, ("repo", "number"))  # Use repo and number as composite key
    except Exception as e:
        print(f"An add_open_closed_issue_node error occurred: {e}")
    return node

def add_open_closed_pr_node(doc, type):
    try:
        properties = {
            'number': doc['number'],
            'created_at': format_datetime(doc['created_at']),
            'repo': doc['name']
        }
        if type == 'OpenPR':
            properties['updated_at'] = format_datetime(doc.get('updated_at', None))
        elif type == 'ClosedPR':
            properties['closed_at'] = format_datetime(doc.get('closed_at', None))
        node = Node(type, **properties)
        graph.merge(node, type, ("repo", "number"))  # Use repo and number as composite key
    except Exception as e:
        print(f"An add_open_closed_pr_node error occurred: {e}")
    return node

# Find node, possibly an OpenPR or ClosedPR node
def find_pr_node(pr_number,repo):
    try:
        query = """
        MATCH (n)
        WHERE (n:OpenPR OR n:ClosedPR) AND n.number = $pr_number AND n.repo = $repo
        RETURN n
        """
        result = graph.run(query, pr_number=pr_number, repo=repo).data()
        if result:
            return result[0]['n']
    except Exception as e:
        print(f"An find_pr_node error occurred: {e}")
    return None

# Handle events
# For simplicity, just to display collaborative actions, specific content not included yet, can be extended later.
def handle_issue_events(issue_node, events):
    if issue_node is None:
        print("Error: issue_node is None. Skipping event handling.")
        return

    tx = graph.begin()
    try:
        for event in events:
            properties = {'created_time': format_datetime(event['time'])} 
            rel_type = event['type'].upper()
            if event['type'] == 'assigned':
                user_node = add_user_node(event['assignee']) if event.get('assignee') else None
                if user_node:
                    graph.create(Relationship(issue_node, rel_type, user_node, **properties))
            elif event['type'] == 'labeled':
                user_node = add_user_node(event['actor']) if event.get('actor') else None
                if user_node:
                    graph.create(Relationship(user_node, rel_type, issue_node, **properties))
            elif event['type'] == 'commented':
                user_node = add_user_node(event['actor']) if event.get('actor') else None
                if user_node:
                    graph.create(Relationship(user_node, rel_type, issue_node, **properties))
            elif event['type'] == 'cross-referenced' and 'source_number' in event:
                target_pr_node = find_pr_node(event['source_number'],issue_node['repo'])
                if target_pr_node:
                    graph.create(Relationship(issue_node, rel_type, target_pr_node, **properties))
    except Exception as e:
        print(f"An handle_issue_events error occurred: {e}")
    graph.commit(tx)

def handle_pr_events(pr_node,events):
    tx = graph.begin()
    try:
        for event in events:
            properties = {'created_time': format_datetime(event['time'])}
            rel_type = event['type'].upper()
            user_node = add_user_node(event['actor']) if event.get('actor') else None
            if user_node:
                graph.create(Relationship(user_node, rel_type, pr_node, **properties))
    except Exception as e:
        print(f"An handle_pr_events error occurred: {e}")
    graph.commit(tx)

# Import data
# Since issues and PRs have referencing relationships, we focus on the PR resolving issue references here, so PRs need to be created first
db_collection = ['open_issues', 'resolved_issues', 'open_prs', 'closed_prs']
db_collection = ['open_issues', 'resolved_issues', 'open_prs', 'closed_prs']
def import_data():  
    tx = graph.begin()
    query_conditions = {"owner": config['repo']['owner'], "name": config['repo']['name']}
    print("Owner:", config['repo']['owner'])
    print("Name:", config['repo']['name'])
    print("Query conditions:", query_conditions)
    print("begin open_pr")
    try:
        i = 0
        for doc in db[db_collection[2]].find(query_conditions):
            properties = {'created_time': format_datetime(doc['created_at'])}
            node_type = 'OpenPR'
            node = add_open_closed_pr_node(doc, node_type)  # open pr
            opener = add_user_node(doc['pr_opener'])  # open pr opener
            graph.create(Relationship(opener, "OPENED", node, **properties))
            handle_pr_events(node, doc['reviewer_events'])  # open pr events
            handle_pr_events(node, doc['normal_commenter_events'])  # open pr events
            handle_pr_events(node, doc['label_events'])  # open pr events
            i += 1
            if i % 1000 == 0:
                print("1000 openpr update!!!")
    except Exception as e:
        print(f"An import_data error occurred: {e}")
        graph.rollback(tx)
    print("begin closed_pr")
    try:
        i = 0
        for doc in db[db_collection[3]].find(query_conditions):
            properties = {'created_time': format_datetime(doc['created_at'])}
            node_type = 'ClosedPR'
            node = add_open_closed_pr_node(doc, node_type)  # closed pr
            opener = add_user_node(doc['pr_opener'])  # closed pr opener
            graph.create(Relationship(opener, "OPENED", node, **properties))
            handle_pr_events(node, doc['reviewer_events'])  # closed pr events
            handle_pr_events(node, doc['normal_commenter_events'])  # closed pr events
            handle_pr_events(node, doc['label_events'])  # closed pr events
            i += 1
            if i % 1000 == 0:
                print("1000 ClosedPR update!!!")
    except Exception as e:
        print(f"An import_data error occurred: {e}")
        graph.rollback(tx)
    print("begin open_issue")
    try:
        i = 0
        for doc in db[db_collection[0]].find(query_conditions):
            properties = {'created_time': format_datetime(doc['created_at'])}
            node_type = 'OpenIssue'
            node = add_open_closed_issue_node(doc, node_type)  # open issue
            opener = add_user_node(doc['issue_opener'])  # open issue opener
            graph.create(Relationship(opener, "OPENED", node, **properties))
            handle_issue_events(node, doc['events'])  # open issue events
            i += 1
            if i % 1000 == 0:
                print("1000 OpenIssue update!!!")
    except Exception as e:
        print(f"An import_data error occurred: {e}")
        graph.rollback(tx)
    print("begin resolved_issue")
    try:
        i = 0
        for doc in db[db_collection[1]].find(query_conditions):
            properties = {'created_time': format_datetime(doc['created_at'])}
            node_type = 'ResolvedIssue'
            node = add_open_closed_issue_node(doc, node_type)  # resolved issue
            opener = add_user_node(doc['issue_opener'])  # resolved issue opener
            graph.create(Relationship(opener, "OPENED", node, **properties))
            handle_issue_events(node, doc['events'])  # resolved issue events
            i += 1
            if i % 1000 == 0:
                print("1000 ResolvedIssue update!!!")
    except Exception as e:
        print(f"An import_data error occurred: {e}")
        graph.rollback(tx)

    graph.commit(tx)

import_data()
print("Data import complete.")