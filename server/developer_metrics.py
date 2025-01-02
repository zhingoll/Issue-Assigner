from datetime import datetime, timezone
from pymongo import MongoClient
import requests
from clickhouse_driver import Client
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ModelConf
import math

print("Reading ClickHouse configurations...")
c_conf = ModelConf("clickhouse.conf")
owner = c_conf['owner']
name = c_conf['name']

print("Connecting to MongoDB...")
client = MongoClient(c_conf['mongodb_uri'])
db = client[c_conf['mongodb_db']]

resolved_issues = db["resolved_issues"]
developer_avg_response = db["developer_metrics"]

print("Clearing old data in developer_avg_response...")
delete_result = developer_avg_response.delete_many({"owner": owner, "name": name})
print(f"Deleted {delete_result.deleted_count} documents.")

response_times = {}

print("Fetching all resolved issues from MongoDB...")
all_issues = list(resolved_issues.find({"owner": owner, "name": name}))
print(f"Total issues fetched: {len(all_issues)}")

print("Collecting all developers involved...")
all_developers = set()
for issue in all_issues:
    events = issue.get("events", [])
    resolvers = issue.get("resolver", [])
    all_developers.add(issue["issue_opener"])
    for ev in events:
        actor = ev.get("actor")
        if actor:
            all_developers.add(actor)
    for r in resolvers:
        all_developers.add(r)

print(f"Total distinct developers: {len(all_developers)}")
print("Developers list sample:", list(all_developers)[:10])

print("Calculating response times for each issue...")
for i, issue in enumerate(all_issues, start=1):
    created_at = issue["created_at"]
    events = issue.get("events", [])

    issue_response_time = {dev: 0 for dev in all_developers}
    dev_first_event_time = {}
    for ev in events:
        actor = ev.get("actor")
        if actor and actor in all_developers:
            event_time = ev["time"]
            if actor not in dev_first_event_time:
                dev_first_event_time[actor] = event_time

    for dev in all_developers:
        if dev in dev_first_event_time:
            delta = dev_first_event_time[dev] - created_at
            issue_response_time[dev] = delta.total_seconds()
        else:
            issue_response_time[dev] = 0

    for dev, rt in issue_response_time.items():
        if dev not in response_times:
            response_times[dev] = []
        response_times[dev].append(rt)

    if i % 50 == 0:
        print(f"Processed {i}/{len(all_issues)} issues...")

print("Calculating average response time per developer...")
dev_avg_response = {}
for dev, times in response_times.items():
    dev_avg_response[dev] = sum(times) / len(times) if times else 0

print("Average response time computed. Sample:")
for d in list(dev_avg_response.keys())[:5]:
    print(d, dev_avg_response[d])

# ------------------ Activity data processing (with index weighting) ---------------------
print("Fetching activity data from remote URL...")
activity_url = f"https://oss.open-digger.cn/github/{owner}/{name}/activity_details.json"
resp = requests.get(activity_url)
if resp.status_code == 200:
    activity_data = resp.json()
    print("Activity data fetched successfully.")
else:
    activity_data = {}
    print("Failed to fetch activity data, using empty data.")

def get_last_three_months(date):
    months = []
    y = date.year
    m = date.month
    for i in range(3):
        yy = y
        mm = m - i
        if mm <= 0:
            yy -= 1
            mm += 12
        months.append(f"{yy:04d}-{mm:02d}")
    return months

threshold_date = datetime(2024, 12, 30)
months_to_consider = get_last_three_months(threshold_date)
print(f"Months considered for activity data: {months_to_consider}")

print("Months available in activity_data:", list(activity_data.keys())[:10])
for m in months_to_consider:
    md = activity_data.get(m, [])
    print(f"Month {m} data length:", len(md))
    if md:
        print("Sample data for", m, md[:3])

tau = 1.0
def exponential_weight(i):
    return math.exp(-(i-1)/tau)

dev_activity_weightsum = {dev: 0.0 for dev in all_developers}
dev_activity_weighted_sum = {dev: 0.0 for dev in all_developers}

print("Calculating weighted activity for each developer...")
for i, m in enumerate(months_to_consider, start=1):
    monthly_data = activity_data.get(m, [])
    month_activity_map = {}
    for item in monthly_data:
        d, val = item
        month_activity_map[d] = month_activity_map.get(d, 0) + val

    w = exponential_weight(i)

    print(f"{m}: dev_count_in_activity_data = {len(month_activity_map)}")

    for dev in all_developers:
        val = month_activity_map.get(dev, 0)
        # If val=0, it means there is no data and it is not included in the numerator and denominator
        if val > 0:
            dev_activity_weightsum[dev] += w
            dev_activity_weighted_sum[dev] += w * val

sample_devs = list(all_developers)[:10]
for sd in sample_devs:
    print(f"Dev: {sd}, activity_weightsum: {dev_activity_weightsum[sd]}, activity_weighted_sum: {dev_activity_weighted_sum[sd]}")

dev_avg_activity = {}
for dev in all_developers:
    if dev_activity_weightsum[dev] > 0:
        dev_avg_activity[dev] = dev_activity_weighted_sum[dev] / dev_activity_weightsum[dev]
    else:
        dev_avg_activity[dev] = 0.0

print("Weighted average activity computed. Sample:")
for d in sample_devs:
    print(d, dev_avg_activity[d])

# ------------------------ Openrank Data Processing ------------------------
# Using a 3-month simple average for community_openrank and globalis_openrank

print("Connecting to ClickHouse...")
ch_client = Client(
    host=c_conf['host'],
    user=c_conf['user'],
    password=c_conf['password'],
    database=c_conf['database'],
    port=c_conf['port']
)

repo_full_name = f"{owner}/{name}"

def parse_year_month(ym_str):
    y, m = ym_str.split('-')
    y = int(y)
    m = int(m)
    start = datetime(y, m, 1)
    if m == 12:
        next_start = datetime(y+1, 1, 1)
    else:
        next_start = datetime(y, m+1, 1)
    return start, next_start

print("Querying community_openrank data for each month...")
community_monthly_maps = []
for m_str in months_to_consider:
    start, next_start = parse_year_month(m_str)
    start_str = start.strftime('%Y-%m-%d %H:%M:%S')
    end_str = next_start.strftime('%Y-%m-%d %H:%M:%S')

    community_query = f"""
    SELECT actor_login, AVG(openrank) as avg_openrank
    FROM opensource.community_openrank
    WHERE platform = 'GitHub'
      AND repo_name = '{repo_full_name}'
      AND created_at >= '{start_str}'
      AND created_at < '{end_str}'
    GROUP BY actor_login
    """
    c_data = ch_client.execute(community_query)
    print(f"{m_str} community data length: {len(c_data)}")
    if c_data:
        print("Sample community_data:", c_data[:5])
    c_map = {row[0]: row[1] for row in c_data}
    community_monthly_maps.append(c_map)

print("Querying global_openrank data for each month...")
developers_list = ",".join([f"'{dev}'" for dev in all_developers])
global_monthly_maps = []
for m_str in months_to_consider:
    start, next_start = parse_year_month(m_str)
    start_str = start.strftime('%Y-%m-%d %H:%M:%S')
    end_str = next_start.strftime('%Y-%m-%d %H:%M:%S')

    global_query = f"""
    SELECT actor_login, AVG(openrank) as avg_openrank
    FROM opensource.global_openrank
    WHERE platform = 'GitHub'
      AND actor_login IN ({developers_list})
      AND type = 'User'
      AND created_at >= '{start_str}'
      AND created_at < '{end_str}'
    GROUP BY actor_login
    """
    g_data = ch_client.execute(global_query)
    print(f"{m_str} global data length: {len(g_data)}")
    if g_data:
        print("Sample global_data:", g_data[:5])
    g_map = {row[0]: row[1] for row in g_data}
    global_monthly_maps.append(g_map)

dev_community_openrank = {}
dev_global_openrank = {}

for dev in sample_devs:
    print(f"Before averaging, checking {dev}:")

for dev in all_developers:
    community_values = []
    for c_map in community_monthly_maps:
        val = c_map.get(dev, 0)
        community_values.append(val)

    if len(community_values) > 0:
        dev_community_openrank[dev] = sum(community_values)/len(community_values)
    else:
        dev_community_openrank[dev] = 0

    global_values = []
    for g_map in global_monthly_maps:
        val = g_map.get(dev, 0)
        global_values.append(val)

    if len(global_values) > 0:
        dev_global_openrank[dev] = sum(global_values)/len(global_values)
    else:
        dev_global_openrank[dev] = 0

for sd in sample_devs:
    print(sd, "community_openrank:", dev_community_openrank[sd], "global_openrank:", dev_global_openrank[sd], "avg_activity:", dev_avg_activity[sd], "avg_response_time:", dev_avg_response.get(sd,0))

print("Preparing documents for MongoDB insertion...")
current_time = datetime.now(timezone.utc)
response_docs = []
for dev in all_developers:
    response_docs.append({
        "owner": owner,
        "name": name,
        "developer": dev,
        "avg_response_time": dev_avg_response.get(dev, 0),
        "avg_activity": dev_avg_activity.get(dev, 0),
        "community_openrank": dev_community_openrank.get(dev, 0),
        "global_openrank": dev_global_openrank.get(dev, 0),
        "update_time": current_time
    })

print(f"Prepared {len(response_docs)} documents for insertion. Sample:")
print(response_docs[:3])

print("Inserting documents into MongoDB...")
insert_result = developer_avg_response.insert_many(response_docs)
print(f"Inserted {len(insert_result.inserted_ids)} documents.")

print("Data calculation and storage are completed.")

