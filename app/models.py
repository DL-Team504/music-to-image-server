from sqlalchemy import Column, Integer, String

from .database import Base


class GeneratedImage(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, index=True)
    title = Column(String)
    creation_date = Column(String)
