from sqlalchemy import Column, Integer, String
from db import Base

class Records(Base):
    __tablename__ = "records"

    id = Column(Integer, primary_key=True, index=True)
    
    # stores each classification the API does
    entries = Column(String, index=True)
