from app.database import Base
from sqlalchemy import Column, Integer, String


class User(Base):
    #id: Optional[int] = Field(default=None, primary_key=True)
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True, unique=True)
    hashed_password = Column(String)

