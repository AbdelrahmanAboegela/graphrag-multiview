"""Graph expansion for multi-view knowledge retrieval."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase
import asyncio


class GraphContext(BaseModel):
    """Expanded graph context from traversal."""
    facts: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    paths: List[Dict[str, Any]]
    total_hops: int


class GraphExpander:
    """Expand context using multi-view graph traversal."""
    
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "graphrag123")
        )
    
    async def expand(
        self,
        seed_chunks: List[str],
        intent: str,
        max_hops: int = 2
    ) -> GraphContext:
        """Expand context from seed chunks using graph traversal.
        
        Args:
            seed_chunks: List of chunk IDs from vector search
            intent: Query intent (procedure, troubleshooting, etc.)
            max_hops: Maximum traversal depth
            
        Returns:
            GraphContext with expanded facts and nodes
        """
        
        # Intent-specific traversal strategies
        traversal_configs = {
            "procedure": {
                "edge_types": ["APPLIES_TO", "HAS_COMPONENT", "REQUIRES"],
                "node_types": ["Asset", "Component", "Concept"]
            },
            "troubleshooting": {
                "edge_types": ["AFFECTS", "CAUSED_BY", "INDICATES", "HAS_COMPONENT"],
                "node_types": ["Concept", "Component", "Asset", "FailureMode"]
            },
            "safety": {
                "edge_types": ["SAFETY_OVERSIGHT", "APPLIES_TO"],
                "node_types": ["Role", "Person", "Asset"]
            },
            "asset_info": {
                "edge_types": ["HAS_COMPONENT", "LOCATED_AT", "RESPONSIBLE_FOR"],
                "node_types": ["Asset", "Component", "Location", "Role"]
            },
            "people": {
                "edge_types": ["HAS_ROLE", "RESPONSIBLE_FOR", "PERFORMED"],
                "node_types": ["Person", "Role", "Asset", "MaintenanceEvent"]
            }
        }
        
        config = traversal_configs.get(intent, traversal_configs["procedure"])
        
        async with self.driver.session() as session:
            facts = []
            all_nodes = []
            all_paths = []
            
            # For each seed chunk, traverse the graph
            for chunk_id in seed_chunks[:5]:  # Limit to top 5 chunks
                try:
                    # Find entities mentioned in chunk
                    result = await session.run("""
                        MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS]->(entity)
                        RETURN entity, labels(entity) as labels
                        LIMIT 5
                    """, {"chunk_id": chunk_id})
                    
                    entities = await result.data()
                    
                    if not entities:
                        continue
                    
                    # For each entity, find its connections
                    for entity_record in entities:
                        entity = entity_record["entity"]
                        entity_labels = entity_record["labels"]
                        entity_id = entity.get("id")
                        entity_name = entity.get("name", entity_id)
                        
                        if not entity_id:
                            continue
                        
                        entity_label = entity_labels[0] if entity_labels else "Entity"
                        
                        # Special handling for people queries - do 2-hop traversal
                        if intent == "people" and entity_label == "Asset":
                            # For people queries starting from Asset, find Role -> Person path
                            traversal_result = await session.run("""
                                MATCH (a:Asset {id: $asset_id})<-[:RESPONSIBLE_FOR]-(r:Role)<-[:HAS_ROLE]-(p:Person)
                                RETURN p.name as person_name, r.name as role_name, 
                                       p.id as person_id, r.id as role_id
                                LIMIT 5
                            """, {"asset_id": entity_id})
                            
                            people_results = await traversal_result.data()
                            
                            for person_rec in people_results:
                                fact_text = f"{person_rec['person_name']} ({person_rec['role_name']}) is responsible for {entity_name}"
                                
                                facts.append({
                                    "fact": fact_text,
                                    "source_nodes": [person_rec['person_id'], person_rec['role_id'], entity_id],
                                    "confidence": 0.95,
                                    "hops": 2
                                })
                                
                                all_paths.append({
                                    "start": f"Person:{person_rec['person_id']}",
                                    "rel": "HAS_ROLE -> RESPONSIBLE_FOR",
                                    "end": f"Asset:{entity_id}"
                                })
                        
                        # Standard 1-hop traversal with DIRECTED relationships
                        # This prevents semantically wrong facts like "Pump responsible for Person"
                        traversal_result = await session.run(f"""
                            MATCH (start:{entity_label} {{id: $entity_id}})
                            OPTIONAL MATCH (start)-[r_out:HAS_COMPONENT|LOCATED_AT|PERFORMED]->(connected_out)
                            OPTIONAL MATCH (start)<-[r_in:RESPONSIBLE_FOR|HAS_ROLE]-(connected_in)
                            WITH start,
                                 COLLECT({{rel: r_out, node: connected_out}}) + 
                                 COLLECT({{rel: r_in, node: connected_in}}) AS connections
                            UNWIND connections AS conn
                            WITH conn
                            WHERE conn.node IS NOT NULL
                            RETURN type(conn.rel) as rel_type, 
                                   labels(conn.node)[0] as connected_label, 
                                   conn.node.id as connected_id, 
                                   conn.node.name as connected_name
                            LIMIT 10
                        """, {"entity_id": entity_id})
                        
                        connections = await traversal_result.data()
                        
                        # Track seen facts to avoid duplicates
                        seen_facts = set()
                        
                        for conn in connections:
                            rel_type = conn["rel_type"]
                            connected_label = conn["connected_label"]
                            connected_name = conn.get("connected_name") or conn.get("connected_id", "Unknown")
                            
                            # Skip unhelpful facts (like Component -> Chunk)
                            if connected_label == "Chunk":
                                continue
                            
                            # Create simple fact
                            fact_text = f"{entity_name} {rel_type.lower().replace('_', ' ')} {connected_name}"
                            
                            # Skip if duplicate
                            if fact_text in seen_facts:
                                continue
                            
                            seen_facts.add(fact_text)
                            
                            facts.append({
                                "fact": fact_text,
                                "source_nodes": [entity_id, conn.get("connected_id")],
                                "confidence": 0.9,
                                "hops": 1
                            })
                            
                            all_nodes.append(entity)
                            all_paths.append({
                                "start": entity_label + ":" + str(entity_id),
                                "rel": rel_type,
                                "end": connected_label + ":" + str(conn.get("connected_id"))
                            })
                
                except Exception as e:
                    # Log error but continue
                    print(f"Error processing chunk {chunk_id}: {e}")
                    continue
            
            # Deduplicate facts before returning
            unique_facts = []
            seen_fact_texts = set()
            for fact in facts:
                if fact["fact"] not in seen_fact_texts:
                    unique_facts.append(fact)
                    seen_fact_texts.add(fact["fact"])
            
            return GraphContext(
                facts=unique_facts[:30],  # Limit facts
                nodes=all_nodes[:50],  # Limit nodes
                paths=all_paths[:20],  # Limit paths
                total_hops=max_hops
            )
    
    def _create_fact_text(self, start_node: Any, rel_type: str, end_node: Any) -> str:
        """Create human-readable fact from graph path."""
        start_labels = list(start_node.labels)
        end_labels = list(end_node.labels)
        
        start_name = start_node.get("name", start_node.get("id", "Unknown"))
        end_name = end_node.get("name", end_node.get("id", "Unknown"))
        
        # Format based on relationship type
        if rel_type == "HAS_COMPONENT":
            return f"{start_name} has component {end_name}"
        elif rel_type == "LOCATED_AT":
            return f"{start_name} is located at {end_name}"
        elif rel_type == "RESPONSIBLE_FOR":
            return f"{start_name} is responsible for {end_name}"
        elif rel_type == "AFFECTS":
            return f"{start_name} affects {end_name}"
        elif rel_type == "PERFORMED":
            return f"{start_name} performed maintenance on {end_name}"
        elif rel_type == "APPLIES_TO":
            return f"Procedure applies to {end_name}"
        else:
            return f"{start_name} {rel_type.lower().replace('_', ' ')} {end_name}"
    
    async def close(self):
        """Close Neo4j connection."""
        await self.driver.close()
