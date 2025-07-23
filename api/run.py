import sys
import os
from pathlib import Path

# 添加项目根目录和 api 目录到 sys.path
project_root = Path(__file__).parent.parent
api_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(api_dir))

from api.main import app
from uvicorn import Config, Server
import signal


# 修改日志路径的计算方式
#log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'api.log')
try:
    with open(log_file, 'a') as f:
        pass
except Exception as e:
    raise e

logging_config = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "level": "DEBUG",
        },
        "file": {
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 30,      # 备份文件
            "level": "DEBUG",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        },
        "uvicorn": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        },
        "uvicorn.error": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.access": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    },
}

def api_run():
    config = Config(
        app=app,
        host="0.0.0.0",
        port=23333,
        log_config=logging_config,
        reload=False,
        workers=1  # 改为1，避免多进程冲突，适合GPU密集型应用
    )
    server = Server(config=config)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    server.run()

def handle_shutdown(signum, frame):
    print("Shutdown initiated")

if __name__ == "__main__":
    api_run()

