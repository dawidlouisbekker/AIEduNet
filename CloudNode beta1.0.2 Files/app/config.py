import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = "sqlite:///./database.sqlite3"        #os.getenv("DATABASE_URL", "sqlite:///./database.db")
SECRET_KEY =  'iox-dIJ-928P2F0[W9JAS]'             #os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
