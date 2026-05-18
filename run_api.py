import sys
sys.path.insert(0, 'src')
import uvicorn
uvicorn.run("cineiq.api.app:app", host="0.0.0.0", port=8000)
