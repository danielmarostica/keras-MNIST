from sqlalchemy import Column, Integer, String
from db import Base

class Records(Base):
    __tablename__ = "records"

    id = Column(Integer, primary_key=True, index=True)
    entries = Column(String, index=True)
