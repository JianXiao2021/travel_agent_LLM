import yaml
from datetime import datetime, timezone, timedelta
import csv
import os

def get_current_local_datetime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

def get_thread_id():
    return get_current_local_datetime().replace("-", "_").replace(":", "")

def save_chat_history(app, config):
    messages = app.get_state(config).values['chat_history']
    thread_id = config['configurable']['thread_id']
    folder_path = os.path.join(os.getcwd(), 'history')
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{thread_id}.csv')

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(messages)

    return file_path