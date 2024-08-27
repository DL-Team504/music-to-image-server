from pydantic import BaseModel, ConfigDict

# class GeneratedImage(Base):
#     __tablename__ = "images"

#     id = Column(Integer, primary_key=True)
#     path = Column(String, unique=True, index=True)
#     title = Column(String)
#     creation_date = Column(String)


class GalleryImageBase(BaseModel):
    path: str
    title: str
    creation_date: str


class GalleryImageCreate(GalleryImageBase):
    pass


class GalleryImage(GalleryImageBase):
    id: int

    model_config = ConfigDict(from_attributes=True)
