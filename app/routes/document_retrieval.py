"""
Document Retrieval API Endpoints
Dedicated endpoints for retrieving specific document data fields
Each endpoint supports lookup by either database ID or UUID
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from app.database.database import get_db
from app.database.models import Document

# Create router for document retrieval endpoi
router = APIRouter(
    prefix="/api/documents",
    tags=["Document Retrieval"]
)

def get_document_by_identifier(document_identifier: str, db: Session) -> Document:
    """
    Helper function to retrieve document by ID or UUID
    
    Args:
        document_identifier: Either integer ID or UUID string
        db: Database session
    
    Returns:
        Document instance
    
    Raises:
        HTTPException: If document not found
    """
    try:
        # Try to parse as integer ID first
        doc_id = int(document_identifier)
        document = db.query(Document).filter(Document.id == doc_id).first()
    except ValueError:
        # If not an integer, treat as UUID
        document = db.query(Document).filter(
            Document.document_uuid == document_identifier
        ).first()
    
    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found with identifier: {document_identifier}"
        )
    
    return document


@router.get("/{document_identifier}/buyer")
async def get_buyer(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get buyer information for a specific document
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Buyer name and document metadata
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "buyer": document.buyer,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/seller")
async def get_seller(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get seller information for a specific document
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Seller name and document metadata
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "seller": document.seller,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/deadlines")
async def get_deadlines(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get all deadlines from a specific document
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        List of deadlines with document metadata
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "deadlines": document.deadlines or [],
        "deadline_count": len(document.deadlines) if document.deadlines else 0,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/obligations")
async def get_obligations(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get all obligations from a specific document
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        List of obligations with document metadata
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "obligations": document.obligations or [],
        "obligation_count": len(document.obligations) if document.obligations else 0,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/alerts")
async def get_alerts(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get all alerts from a specific document
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        List of alerts with document metadata
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "alerts": document.alerts or [],
        "alert_count": len(document.alerts) if document.alerts else 0,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/summary")
async def get_summary(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get document summary
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Document summary and metadata
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "document_type": document.document_type,
        "summary": document.summary,
        "page_count": document.page_count,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/cleaned-text")
async def get_cleaned_text(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get cleaned extracted text from document
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Cleaned text content
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "cleaned_text": document.cleaned_text,
        "text_length": len(document.cleaned_text) if document.cleaned_text else 0,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/parties")
async def get_parties(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get both buyer and seller information
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Buyer and seller information
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "buyer": document.buyer,
        "seller": document.seller,
        "retrieved_at": document.created_at.isoformat()
    }


@router.get("/{document_identifier}/key-info")
async def get_key_info(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get all key information in one call (consolidated endpoint)
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Comprehensive key information including buyer, seller, deadlines, obligations, alerts
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "document_type": document.document_type,
        "buyer": document.buyer,
        "seller": document.seller,
        "summary": document.summary,
        "deadlines": {
            "items": document.deadlines or [],
            "count": len(document.deadlines) if document.deadlines else 0
        },
        "obligations": {
            "items": document.obligations or [],
            "count": len(document.obligations) if document.obligations else 0
        },
        "alerts": {
            "items": document.alerts or [],
            "count": len(document.alerts) if document.alerts else 0
        },
        "metadata": {
            "page_count": document.page_count,
            "extraction_method": document.extraction_method,
            "processing_time_seconds": document.processing_time_seconds,
            "created_at": document.created_at.isoformat()
        }
    }


@router.get("/{document_identifier}/metadata")
async def get_metadata(
    document_identifier: str,
    db: Session = Depends(get_db)
):
    """
    Get document processing metadata
    
    Args:
        document_identifier: Document ID or UUID
    
    Returns:
        Document metadata including processing information
    """
    document = get_document_by_identifier(document_identifier, db)
    
    return {
        "document_id": document.id,
        "document_uuid": document.document_uuid,
        "document_name": document.document_name,
        "document_type": document.document_type,
        "page_count": document.page_count,
        "extraction_method": document.extraction_method,
        "processing_time_seconds": document.processing_time_seconds,
        "created_at": document.created_at.isoformat(),
        "has_buyer": document.buyer is not None,
        "has_seller": document.seller is not None,
        "has_summary": document.summary is not None,
        "deadline_count": len(document.deadlines) if document.deadlines else 0,
        "obligation_count": len(document.obligations) if document.obligations else 0,
        "alert_count": len(document.alerts) if document.alerts else 0
    }


# ============================================================================
# BATCH RETRIEVAL ENDPOINTS
# ============================================================================

@router.get("/batch/deadlines")
async def get_all_deadlines(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum records to return"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    db: Session = Depends(get_db)
):
    """
    Get deadlines from multiple documents
    
    Args:
        skip: Pagination offset
        limit: Maximum number of documents
        document_type: Optional filter by document type
    
    Returns:
        List of documents with their deadlines
    """
    query = db.query(Document).filter(Document.deadlines.isnot(None))
    
    if document_type:
        query = query.filter(Document.document_type == document_type)
    
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for doc in documents:
        results.append({
            "document_id": doc.id,
            "document_uuid": doc.document_uuid,
            "document_name": doc.document_name,
            "deadlines": doc.deadlines,
            "deadline_count": len(doc.deadlines) if doc.deadlines else 0
        })
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": results
    }


@router.get("/batch/obligations")
async def get_all_obligations(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    document_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get obligations from multiple documents
    """
    query = db.query(Document).filter(Document.obligations.isnot(None))
    
    if document_type:
        query = query.filter(Document.document_type == document_type)
    
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for doc in documents:
        results.append({
            "document_id": doc.id,
            "document_uuid": doc.document_uuid,
            "document_name": doc.document_name,
            "obligations": doc.obligations,
            "obligation_count": len(doc.obligations) if doc.obligations else 0
        })
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": results
    }


@router.get("/batch/alerts")
async def get_all_alerts(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    document_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get alerts from multiple documents
    """
    query = db.query(Document).filter(Document.alerts.isnot(None))
    
    if document_type:
        query = query.filter(Document.document_type == document_type)
    
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for doc in documents:
        results.append({
            "document_id": doc.id,
            "document_uuid": doc.document_uuid,
            "document_name": doc.document_name,
            "alerts": doc.alerts,
            "alert_count": len(doc.alerts) if doc.alerts else 0
        })
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": results
    }


@router.get("/search/by-buyer")
async def search_by_buyer(
    buyer_name: str = Query(..., description="Buyer name to search for"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Search documents by buyer name (case-insensitive partial match)
    """
    query = db.query(Document).filter(
        Document.buyer.ilike(f"%{buyer_name}%")
    )
    
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for doc in documents:
        results.append({
            "document_id": doc.id,
            "document_uuid": doc.document_uuid,
            "document_name": doc.document_name,
            "buyer": doc.buyer,
            "seller": doc.seller,
            "document_type": doc.document_type,
            "created_at": doc.created_at.isoformat()
        })
    
    return {
        "search_term": buyer_name,
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": results
    }


@router.get("/search/by-seller")
async def search_by_seller(
    seller_name: str = Query(..., description="Seller name to search for"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Search documents by seller name (case-insensitive partial match)
    """
    query = db.query(Document).filter(
        Document.seller.ilike(f"%{seller_name}%")
    )
    
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for doc in documents:
        results.append({
            "document_id": doc.id,
            "document_uuid": doc.document_uuid,
            "document_name": doc.document_name,
            "buyer": doc.buyer,
            "seller": doc.seller,
            "document_type": doc.document_type,
            "created_at": doc.created_at.isoformat()
        })
    
    return {
        "search_term": seller_name,
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": results
    }