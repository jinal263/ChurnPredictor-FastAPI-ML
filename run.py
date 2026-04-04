"""
run.py — start the server.

Usage:
  python run.py  

Or directly with uvicorn:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
