# flask_server.py
from flask import Flask, request, jsonify
from celery import Celery
import time

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

@celery.task(bind=True)
def long_running_task(self, param):
    try:
        for i in range(1440):  # 1분마다 작업을 수행, 24시간 동안 총 1440번
            time.sleep(60)  # 1분 대기
            self.update_state(state='PROGRESS', meta={'current': i, 'total': 1440})
            print(f"Task running... {i+1}/1440")
        return {'status': 'Task Completed!', 'result': 'success'}
    except Exception as e:
        return {'status': 'Task Failed!', 'result': str(e)}

@app.route('/start_task', methods=['POST'])
def start_task():
    param = request.json.get('param')
    task = long_running_task.apply_async(args=[param])
    return jsonify({"task_id": task.id}), 202

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = long_running_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'status': task.info['status']
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
