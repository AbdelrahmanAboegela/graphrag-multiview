"""Enhanced ingestion with consistent IDs across Qdrant and Neo4j."""
import asyncio
from pathlib import Path
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import hashlib
from datetime import datetime


async def ingest_with_consistent_ids():
    """Ingest documents with UUID consistency across Qdrant and Neo4j."""
    
    # Initialize
    neo4j_driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "graphrag123")
    )
    
    qdrant_client = QdrantClient(host="localhost", port=6333)
    embedding_model = SentenceTransformer('intfloat/e5-large-v2')
    
    # Recreate Qdrant collection with consistent IDs
    collection_name = "graphrag_chunks"
    
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")
    except:
        pass
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print(f"✓ Created collection: {collection_name}")
    
    # Get all documents
    docs_dir = Path("sample_docs")
    doc_files = list(docs_dir.glob("*.md"))
    
    print(f"\nProcessing {len(doc_files)} documents...")
    
    total_chunks = 0
    
    async with neo4j_driver.session() as session:
        for doc_file in doc_files:
            print(f"\n📄 Processing: {doc_file.name}")
            
            # Read document
            content = doc_file.read_text(encoding='utf-8')
            
            # Create document ID
            doc_id = str(uuid.uuid4())
            
            # Create document node
            await session.run("""
                MERGE (d:Document {id: $id})
                SET d.title = $title,
                    d.source_file = $source_file,
                    d.created_at = datetime()
            """, {
                "id": doc_id,
                "title": doc_file.stem.replace('_', ' ').title(),
                "source_file": doc_file.name
            })
            
            # Split into chunks (by paragraph)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
            
            qdrant_points = []
            
            for i, para in enumerate(paragraphs):
                # Create consistent UUID for chunk
                chunk_uuid = str(uuid.uuid4())
                
                # Create chunk node in Neo4j
                await session.run("""
                    MATCH (d:Document {id: $doc_id})
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $text,
                        position: $position,
                        created_at: datetime()
                    })
                    CREATE (d)-[:CONTAINS]->(c)
                """, {
                    "doc_id": doc_id,
                    "chunk_id": chunk_uuid,
                    "text": para,
                    "position": i
                })
                
                # Generate embedding
                embedding = embedding_model.encode(f"passage: {para}").tolist()
                
                # Create Qdrant point with SAME UUID
                qdrant_points.append(PointStruct(
                    id=chunk_uuid,  # Use UUID string directly
                    vector=embedding,
                    payload={
                        "text": para,
                        "document_id": doc_id,
                        "position": i,
                        "source_file": doc_file.name
                    }
                ))
            
            # Batch upsert to Qdrant
            if qdrant_points:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=qdrant_points
                )
                total_chunks += len(qdrant_points)
                print(f"  ✓ Ingested {len(qdrant_points)} chunks")
    
    print(f"\n✅ Total chunks ingested: {total_chunks}")
    
    # Now create entity links
    print("\n🔗 Creating entity links...")
    
    async with neo4j_driver.session() as session:
        # Get all chunks
        result = await session.run("MATCH (c:Chunk) RETURN c.id as chunk_id, c.text as text")
        chunks = await result.data()
        
        links_created = 0
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = (chunk["text"] or "").lower()
            
            # Link to assets
            if 'p-101' in text or 'pump' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (a:Asset {id: 'P-101'})
                    MERGE (c)-[:MENTIONS]->(a)
                """, {"chunk_id": chunk_id})
                links_created += 1
            
            if 'p-102' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (a:Asset {id: 'P-102'})
                    MERGE (c)-[:MENTIONS]->(a)
                """, {"chunk_id": chunk_id})
                links_created += 1
            
            if 'v-201' in text or 'valve' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (a:Asset {id: 'V-201'})
                    MERGE (c)-[:MENTIONS]->(a)
                """, {"chunk_id": chunk_id})
                links_created += 1
            
            # Link to components
            if 'bearing' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (comp:Component) WHERE comp.type = 'BEARING'
                    MERGE (c)-[:MENTIONS]->(comp)
                """, {"chunk_id": chunk_id})
                links_created += 1
            
            if 'seal' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (comp:Component) WHERE comp.type = 'SEAL'
                    MERGE (c)-[:MENTIONS]->(comp)
                """, {"chunk_id": chunk_id})
                links_created += 1
            
            # Link to roles
            if 'safety' in text or 'ppe' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (r:Role {id: 'safety_officer'})
                    MERGE (c)-[:MENTIONS]->(r)
                """, {"chunk_id": chunk_id})
                links_created += 1
            
            if 'maintenance' in text or 'technician' in text:
                await session.run("""
                    MATCH (c:Chunk {id: $chunk_id})
                    MATCH (r:Role {id: 'mech_tech'})
                    MERGE (c)-[:MENTIONS]->(r)
                """, {"chunk_id": chunk_id})
                links_created += 1
        
        print(f"✓ Created {links_created} MENTIONS relationships")
    
    await neo4j_driver.close()
    qdrant_client.close()
    
    print("\n✅ Ingestion complete with consistent UUIDs!")


if __name__ == "__main__":
    asyncio.run(ingest_with_consistent_ids())
