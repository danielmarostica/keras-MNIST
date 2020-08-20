from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# database connection
SQLALCHEMY_DATABASE_URL = "sqlite:///records.db"

# creates engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# session specs
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# base
Base = declarative_base()
