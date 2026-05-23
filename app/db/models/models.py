import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class MediaFile(Base):
    """جدول الصور والفيديوهات المرفوعة"""
    __tablename__ = "media_files"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename      = Column(String, nullable=False)
    file_path     = Column(String, nullable=False)
    file_hash     = Column(String, nullable=False, unique=True)
    media_type    = Column(String, nullable=False)  # "image" or "video"
    file_size     = Column(Integer, nullable=False)
    width         = Column(Integer, nullable=True)
    height        = Column(Integer, nullable=True)
    duration_sec  = Column(Float, nullable=True)    # للفيديو فقط
    uploaded_at   = Column(DateTime, default=datetime.utcnow)

    # Relationship
    result = relationship("AnalysisResult", back_populates="media", uselist=False)


class AnalysisResult(Base):
    """جدول نتائج التحليل"""
    __tablename__ = "analysis_results"

    id                      = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    media_id                = Column(UUID(as_uuid=True), ForeignKey("media_files.id"))

    # النتيجة النهائية
    final_score             = Column(Float, nullable=False)  # 0.0 - 1.0
    classification          = Column(String, nullable=False) # real/suspicious/likely_ai/ai

    # نتائج كل طبقة
    ml_score                = Column(Float, nullable=True)
    ela_score               = Column(Float, nullable=True)
    fft_score               = Column(Float, nullable=True)
    metadata_score          = Column(Float, nullable=True)
    noise_score             = Column(Float, nullable=True)

    # خاص بالوجوه
    face_detected           = Column(Boolean, default=False)
    face_manipulation_score = Column(Float, nullable=True)

    # خاص بالفيديو
    frames_analyzed         = Column(Integer, nullable=True)
    temporal_score          = Column(Float, nullable=True)

    processing_time_ms      = Column(Integer, nullable=True)
    analyzed_at             = Column(DateTime, default=datetime.utcnow)

    # Relationship
    media = relationship("MediaFile", back_populates="result")