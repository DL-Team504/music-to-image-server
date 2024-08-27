from sqlalchemy.orm import Session

from . import models, schemas


def get_generated_images(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.GeneratedImage).offset(skip).limit(limit).all()


def create_generated_image(
    db: Session, generated_image: schemas.GalleryImageCreate
) -> models.GeneratedImage:
    db_generated_image = models.GeneratedImage(
        path=generated_image.path,
        title=generated_image.title,
        creation_date=generated_image.creation_date,
    )
    db.add(db_generated_image)
    db.commit()
    db.refresh(db_generated_image)

    return db_generated_image
