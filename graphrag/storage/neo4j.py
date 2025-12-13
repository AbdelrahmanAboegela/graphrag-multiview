"""Neo4j Graph Database Client."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from pydantic import SecretStr

from graphrag.core.config import get_settings


class Neo4jClient:
    """Async Neo4j client for graph operations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: SecretStr | None = None,
    ):
        """Initialize Neo4j client.

        Args:
            uri: Neo4j connection URI.
            user: Username.
            password: Password.
        """
        settings = get_settings()
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password.get_secret_value()),
        )
        # Verify connectivity
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @property
    def driver(self) -> AsyncDriver:
        """Get driver instance."""
        if not self._driver:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")
        return self._driver

    @asynccontextmanager
    async def session(self, database: str = "neo4j") -> AsyncIterator[AsyncSession]:
        """Get a database session.

        Args:
            database: Database name.

        Yields:
            Neo4j async session.
        """
        session = self.driver.session(database=database)
        try:
            yield session
        finally:
            await session.close()

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query.

        Args:
            query: Cypher query string.
            parameters: Query parameters.
            database: Target database.

        Returns:
            List of result records as dicts.
        """
        async with self.session(database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str = "neo4j",
    ) -> dict[str, Any]:
        """Execute a write transaction.

        Args:
            query: Cypher query string.
            parameters: Query parameters.
            database: Target database.

        Returns:
            Query summary statistics.
        """
        async with self.session(database) as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "nodes_deleted": summary.counters.nodes_deleted,
                "relationships_created": summary.counters.relationships_created,
                "relationships_deleted": summary.counters.relationships_deleted,
                "properties_set": summary.counters.properties_set,
            }

    # ========================================================================
    # Graph Traversal Methods
    # ========================================================================

    async def get_node_by_id(
        self,
        node_id: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Get a node by its ID property.

        Args:
            node_id: Node ID.
            labels: Optional label filter.

        Returns:
            Node properties or None.
        """
        label_str = ":".join(labels) if labels else ""
        label_clause = f":{label_str}" if label_str else ""

        query = f"""
        MATCH (n{label_clause} {{id: $id}})
        RETURN n
        """
        results = await self.execute_query(query, {"id": node_id})
        return results[0]["n"] if results else None

    async def traverse_from_node(
        self,
        start_node_id: str,
        edge_types: list[str] | None = None,
        max_hops: int = 2,
        direction: str = "BOTH",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Traverse graph from a starting node.

        Args:
            start_node_id: Starting node ID.
            edge_types: Allowed edge types (None = all).
            max_hops: Maximum traversal depth.
            direction: OUTGOING, INCOMING, or BOTH.
            limit: Maximum paths to return.

        Returns:
            List of paths with nodes and relationships.
        """
        edge_filter = "|".join(edge_types) if edge_types else ""
        edge_clause = f":{edge_filter}" if edge_filter else ""

        direction_map = {
            "OUTGOING": f"-[r{edge_clause}*1..{max_hops}]->",
            "INCOMING": f"<-[r{edge_clause}*1..{max_hops}]-",
            "BOTH": f"-[r{edge_clause}*1..{max_hops}]-",
        }
        rel_pattern = direction_map.get(direction, direction_map["BOTH"])

        query = f"""
        MATCH path = (start {{id: $start_id}}){rel_pattern}(end)
        RETURN path, 
               [n IN nodes(path) | properties(n)] AS nodes,
               [r IN relationships(path) | type(r)] AS rel_types
        LIMIT $limit
        """

        results = await self.execute_query(
            query,
            {"start_id": start_node_id, "limit": limit},
        )

        return [
            {
                "nodes": r["nodes"],
                "relationships": r["rel_types"],
            }
            for r in results
        ]

    async def find_paths_between(
        self,
        start_id: str,
        end_id: str,
        max_hops: int = 4,
    ) -> list[dict[str, Any]]:
        """Find paths between two nodes.

        Args:
            start_id: Starting node ID.
            end_id: Ending node ID.
            max_hops: Maximum path length.

        Returns:
            List of paths.
        """
        query = f"""
        MATCH path = shortestPath(
            (start {{id: $start_id}})-[*1..{max_hops}]-(end {{id: $end_id}})
        )
        RETURN path,
               [n IN nodes(path) | properties(n)] AS nodes,
               [r IN relationships(path) | type(r)] AS rel_types
        """

        results = await self.execute_query(
            query,
            {"start_id": start_id, "end_id": end_id},
        )

        return [
            {
                "nodes": r["nodes"],
                "relationships": r["rel_types"],
            }
            for r in results
        ]

    # ========================================================================
    # Entity Operations
    # ========================================================================

    async def merge_entity(
        self,
        entity_type: str,
        entity_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge (upsert) an entity node.

        Args:
            entity_type: Node label.
            entity_id: Entity ID.
            properties: Node properties.

        Returns:
            Write statistics.
        """
        # Remove None values
        props = {k: v for k, v in properties.items() if v is not None}

        query = f"""
        MERGE (n:{entity_type} {{id: $id}})
        SET n += $props
        RETURN n
        """

        return await self.execute_write(query, {"id": entity_id, "props": props})

    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a relationship between two nodes.

        Args:
            from_id: Source node ID.
            to_id: Target node ID.
            rel_type: Relationship type.
            properties: Relationship properties.

        Returns:
            Write statistics.
        """
        props = properties or {}

        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        RETURN r
        """

        return await self.execute_write(
            query,
            {"from_id": from_id, "to_id": to_id, "props": props},
        )

    async def health_check(self) -> bool:
        """Check Neo4j connectivity.

        Returns:
            True if healthy.
        """
        try:
            await self.execute_query("RETURN 1 AS health")
            return True
        except Exception:
            return False


# Singleton instance
_neo4j_client: Neo4jClient | None = None


async def get_neo4j_client() -> Neo4jClient:
    """Get or create Neo4j client singleton."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
        await _neo4j_client.connect()
    return _neo4j_client
