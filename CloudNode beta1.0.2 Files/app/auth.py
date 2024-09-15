from datetime import timedelta,datetime, timezone
from typing import Annotated
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette import status
from app.database import SessionLocal
from app.models import User
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

router = APIRouter(
    prefix='/auth',
    tags=['auth']
)
SECRET_KEY = 'Will be choosen'
ALGORITHM = 'Will be choosen'

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

class CreateUserRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]  
        
@router.post('/oweakdmfaiowejfsdfdsjfeo', status_code=status.HTTP_201_CREATED)
async def create_user(db: db_dependency, create_user_request: CreateUserRequest):
    json_data = await Request.body()
    with open('incoming_json.txt', 'w') as f:
        f.write(json_data)
        
    username = create_user_request.username
    if get_user_by_username(db, username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Username already exists')
    create_user_model= User(username = username,hashed_password = bcrypt_context.hash(create_user_request.password),)
    db.add(create_user_model)
    db.commit()
    return status.HTTP_201_CREATED
   
   
   
@router.post('/token', response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    token = create_user_token(user.username, user.id, timedelta(minutes=20))
    #token = jwt.encode({'sub': user.username, 'exp': datetime.utcnow() + token_expires}, SECRET_KEY)
    return {'access_token': token, 'token_type': 'bearer'}

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def verify_password(plain_password, hashed_password):
    return bcrypt_context.verify(plain_password, hashed_password)

def create_user_token (username: str, userid: int, token_expires: timedelta):
    encode = {'sub': username, 'id': userid}
    expires = datetime.now(timezone.utc) + token_expires
    encode.update({'exp': expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate credentials')
        return {'username': username}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate credentials')
    
router.post('/asijdpawoiefjoaidojkdfpoaij', status_code=status.HTTP_200_OK)
async def user_login(db: db_dependency, create_user_request: CreateUserRequest):
    if authenticate_user(db, create_user_request.username, create_user_request.password):
        return {'access_token': create_user_request.username, 'token_type': 'bearer'}
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Incorrect username or password')

    

