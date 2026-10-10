import uvicorn
from .engine.config import get_config
from .api.routes import app

def main():
    cfg = get_config()
    uvicorn.run(
        "chatterbox.api.routes:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()
