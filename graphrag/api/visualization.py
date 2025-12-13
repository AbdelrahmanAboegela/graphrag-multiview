"""Graph visualization endpoints for monitoring retrieval process."""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pyd antic import BaseModel


router = APIRouter(prefix="/api/v1/graph", tags=["visualization"])


class GraphSnapshot(BaseModel):
    """Snapshot of graph state during retrieval."""
    stage: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Store retrieval snapshots (in production, use Redis or database)
retrieval_snapshots: Dict[str, List[GraphSnapshot]] = {}


@router.get("/query/{query_id}/snapshots")
async def get_query_snapshots(query_id: str) -> List[GraphSnapshot]:
    """Get all graph snapshots for a query's retrieval process."""
    if query_id not in retrieval_snapshots:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    
    return retrieval_snapshots[query_id]


@router.get("/query/{query_id}/snapshot/{stage}")
async def get_query_snapshot_by_stage(query_id: str, stage: str) -> GraphSnapshot:
    """Get graph snapshot for a specific retrieval stage."""
    if query_id not in retrieval_snapshots:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    
    stage_snapshot = next(
        (s for s in retrieval_snapshots[query_id] if s.stage == stage),
        None
    )
    
    if not stage_snapshot:
        raise HTTPException(status_code=404, detail=f"Stage {stage} not found")
    
    return stage_snapshot


@router.get("/stats")
async def get_graph_stats():
    """Get overall graph statistics."""
    from graphrag.storage import get_neo4j_client
    
    neo4j = await get_neo4j_client()
    
    # Get node counts by label
    node_counts_query = """
    MATCH (n)
    RETURN labels(n)[0] as label, count(*) as count
    ORDER BY count DESC
    """
    node_counts = await neo4j.execute_read_query(node_counts_query)
    
    # Get relationship counts by type
    rel_counts_query = """
    MATCH ()-[r]->()
    RETURN type(r) as type, count(*) as count
    ORDER BY count DESC
    """
    rel_counts = await neo4j.execute_read_query(rel_counts_query)
    
    # Get total counts
    total_query = """
    MATCH (n)
    WITH count(n) as nodes
    MATCH ()-[r]->()
    RETURN nodes, count(r) as relationships
    """
    totals_result = await neo4j.execute_read_query(total_query)
    totals = totals_result[0] if totals_result else {"nodes": 0, "relationships": 0}
    
    await neo4j.close()
    
    return {
        "total_nodes": totals.get("nodes", 0),
        "total_relationships": totals.get("relationships", 0),
        "nodes_by_label": {r["label"]: r["count"] for r in node_counts},
        "relationships_by_type": {r["type"]: r["count"] for r in rel_counts}
    }


@router.get("/neighborhood/{node_id}")
async def get_node_neighborhood(node_id: str, hops: int = 2):
    """Get the neighborhood around a specific node."""
    from graphrag.storage import get_neo4j_client
    
    neo4j = await get_neo4j_client()
    
    # Get node and its neighbors up to N hops
    query = f"""
    MATCH path = (root)-[*1..{hops}]-(neighbor)
    WHERE root.id = $node_id OR root.canonical_id = $node_id
    WITH root, neighbor, relationships(path) as rels
    RETURN 
        root,
        collect(DISTINCT neighbor) as neighbors,
        collect(DISTINCT rels) as edges
    LIMIT 1
    """
    
    result = await neo4j.execute_read_query(query, {"node_id": node_id})
    await neo4j.close()
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    record = result[0]
    
    # Format nodes
    nodes = [dict(record["root"])]
    nodes.extend([dict(n) for n in record["neighbors"]])
    
    # Format edges
    edges = []
    for rel_list in record["edges"]:
        for rel in rel_list:
            edges.append({
                "source": rel.start_node.element_id,
                "target": rel.end_node.element_id,
                "type": rel.type,
                "properties": dict(rel)
            })
    
    return {
        "center_node": node_id,
        "hops": hops,
        "nodes": nodes,
        "edges": edges
    }


def capture_retrieval_snapshot(
    query_id: str,
    stage: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    metadata: Dict[str, Any]
):
    """Capture a snapshot of the graph during retrieval (called by retrieval nodes)."""
    snapshot = GraphSnapshot(
        stage=stage,
        nodes=nodes,
        edges=edges,
        metadata=metadata
    )
    
    if query_id not in retrieval_snapshots:
        retrieval_snapshots[query_id] = []
    
    retrieval_snapshots[query_id].append(snapshot)


def clear_old_snapshots(max_queries: int = 100):
    """Clear old snapshots to prevent memory buildup."""
    if len(retrieval_snapshots) > max_queries:
        # Keep only the most recent queries
        sorted_keys = sorted(
            retrieval_snapshots.keys(),
            key=lambda x: retrieval_snapshots[x][-1].metadata.get("timestamp", 0),
            reverse=True
        )
        
        keys_to_remove = sorted_keys[max_queries:]
        for key in keys_to_remove:
            del retrieval_snapshots[key]
