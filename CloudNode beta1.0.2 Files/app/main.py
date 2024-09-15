from fastapi import FastAPI, status, Depends, HTTPException
import app.models as models
from app.database import engine, SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
from app.auth import router, get_current_user


app = FastAPI()
app.include_router(router)

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

@app.get("/", status_code=status.HTTP_200_OK)
async def user(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {"user": user}

