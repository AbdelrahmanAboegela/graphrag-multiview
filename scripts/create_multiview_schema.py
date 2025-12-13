"""Extended Neo4j schema for multi-view GraphRAG."""
from neo4j import AsyncGraphDatabase
import asyncio


async def create_multiview_schema():
    """Create complete multi-view graph schema in Neo4j."""
    
    driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "graphrag123")
    )
    
    async with driver.session() as session:
        print("Creating Multi-View Graph Schema...")
        
        # ============================================================================
        # CONSTRAINTS & INDEXES
        # ============================================================================
        
        constraints = [
            # Existing
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            
            # Asset View
            "CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (a:Asset) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
            
            # People View
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT role_id IF NOT EXISTS FOR (r:Role) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT team_id IF NOT EXISTS FOR (t:Team) REQUIRE t.id IS UNIQUE",
            
            # Temporal View
            "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:MaintenanceEvent) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT inspection_id IF NOT EXISTS FOR (i:Inspection) REQUIRE i.id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                await session.run(constraint)
                print(f"✓ {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"✗ Error: {e}")
        
        # ============================================================================
        # SAMPLE ASSET DATA
        # ============================================================================
        
        print("\nCreating Asset View...")
        
        # Locations
        locations = [
            {"id": "loc_building_a", "name": "Building A", "type": "BUILDING"},
            {"id": "loc_pump_room", "name": "Pump Room", "type": "ROOM", "parent": "loc_building_a"},
            {"id": "loc_valve_station", "name": "Valve Station", "type": "AREA", "parent": "loc_building_a"},
        ]
        
        for loc in locations:
            await session.run("""
                MERGE (l:Location {id: $id})
                SET l.name = $name,
                    l.type = $type,
                    l.created_at = datetime()
            """, loc)
        
        # Link location hierarchy
        await session.run("""
            MATCH (child:Location {id: 'loc_pump_room'})
            MATCH (parent:Location {id: 'loc_building_a'})
            MERGE (child)-[:LOCATED_IN]->(parent)
        """)
        
        await session.run("""
            MATCH (child:Location {id: 'loc_valve_station'})
            MATCH (parent:Location {id: 'loc_building_a'})
            MERGE (child)-[:LOCATED_IN]->(parent)
        """)
        
        print("✓ Created 3 locations")
        
        # Assets
        assets = [
            {
                "id": "P-101",
                "name": "Centrifugal Pump P-101",
                "type": "PUMP",
                "status": "OPERATIONAL",
                "criticality": "A",
                "location_id": "loc_pump_room"
            },
            {
                "id": "P-102",
                "name": "Centrifugal Pump P-102",
                "type": "PUMP",
                "status": "STANDBY",
                "criticality": "A",
                "location_id": "loc_pump_room"
            },
            {
                "id": "V-201",
                "name": "Control Valve V-201",
                "type": "VALVE",
                "status": "OPERATIONAL",
                "criticality": "B",
                "location_id": "loc_valve_station"
            },
        ]
        
        for asset in assets:
            await session.run("""
                MERGE (a:Asset {id: $id})
                SET a.name = $name,
                    a.type = $type,
                    a.status = $status,
                    a.criticality = $criticality,
                    a.created_at = datetime()
            """, asset)
            
            # Link to location
            await session.run("""
                MATCH (a:Asset {id: $asset_id})
                MATCH (l:Location {id: $location_id})
                MERGE (a)-[:LOCATED_AT]->(l)
            """, {"asset_id": asset["id"], "location_id": asset["location_id"]})
        
        print("✓ Created 3 assets")
        
        # Components
        components = [
            {"id": "P-101-BRG", "name": "Bearing Assembly", "type": "BEARING", "parent": "P-101"},
            {"id": "P-101-SEAL", "name": "Mechanical Seal", "type": "SEAL", "parent": "P-101"},
            {"id": "P-101-IMP", "name": "Impeller", "type": "IMPELLER", "parent": "P-101"},
            {"id": "P-102-BRG", "name": "Bearing Assembly", "type": "BEARING", "parent": "P-102"},
            {"id": "P-102-SEAL", "name": "Mechanical Seal", "type": "SEAL", "parent": "P-102"},
            {"id": "V-201-ACT", "name": "Pneumatic Actuator", "type": "ACTUATOR", "parent": "V-201"},
        ]
        
        for comp in components:
            await session.run("""
                MERGE (c:Component {id: $id})
                SET c.name = $name,
                    c.type = $type,
                    c.created_at = datetime()
            """, comp)
            
            # Link to asset
            await session.run("""
                MATCH (a:Asset {id: $parent})
                MATCH (c:Component {id: $comp_id})
                MERGE (a)-[:HAS_COMPONENT]->(c)
            """, {"parent": comp["parent"], "comp_id": comp["id"]})
        
        print("✓ Created 6 components")
        
        # ============================================================================
        # SAMPLE PEOPLE DATA
        # ============================================================================
        
        print("\nCreating People View...")
        
        # Roles
        roles = [
            {"id": "mech_tech", "name": "Mechanical Technician", "level": "L2", "department": "Maintenance"},
            {"id": "safety_officer", "name": "Safety Officer", "level": "L3", "department": "HSE"},
            {"id": "maint_supervisor", "name": "Maintenance Supervisor", "level": "L4", "department": "Maintenance"},
        ]
        
        for role in roles:
            await session.run("""
                MERGE (r:Role {id: $id})
                SET r.name = $name,
                    r.level = $level,
                    r.department = $department,
                    r.created_at = datetime()
            """, role)
        
        print("✓ Created 3 roles")
        
        # Teams
        teams = [
            {"id": "team_rotating", "name": "Rotating Equipment Team", "department": "Maintenance"},
            {"id": "team_safety", "name": "Safety Team", "department": "HSE"},
        ]
        
        for team in teams:
            await session.run("""
                MERGE (t:Team {id: $id})
                SET t.name = $name,
                    t.department = $department,
                    t.created_at = datetime()
            """, team)
        
        print("✓ Created 2 teams")
        
        # People (anonymized)
        people = [
            {"id": "person_001", "name": "John Smith", "department": "Maintenance"},
            {"id": "person_002", "name": "Jane Doe", "department": "HSE"},
            {"id": "person_003", "name": "Bob Wilson", "department": "Maintenance"},
        ]
        
        for person in people:
            await session.run("""
                MERGE (p:Person {id: $id})
                SET p.name = $name,
                    p.department = $department,
                    p.created_at = datetime()
            """, person)
        
        print("✓ Created 3 people")
        
        # Assign roles
        role_assignments = [
            ("person_001", "mech_tech"),
            ("person_002", "safety_officer"),
            ("person_003", "maint_supervisor"),
        ]
        
        for person_id, role_id in role_assignments:
            await session.run("""
                MATCH (p:Person {id: $person_id})
                MATCH (r:Role {id: $role_id})
                MERGE (p)-[:HAS_ROLE]->(r)
            """, {"person_id": person_id, "role_id": role_id})
        
        # Assign to teams
        team_assignments = [
            ("person_001", "team_rotating"),
            ("person_003", "team_rotating"),
            ("person_002", "team_safety"),
        ]
        
        for person_id, team_id in team_assignments:
            await session.run("""
                MATCH (p:Person {id: $person_id})
                MATCH (t:Team {id: $team_id})
                MERGE (p)-[:MEMBER_OF]->(t)
            """, {"person_id": person_id, "team_id": team_id})
        
        # Role responsibilities for assets
        await session.run("""
            MATCH (r:Role {id: 'mech_tech'})
            MATCH (a:Asset) WHERE a.type = 'PUMP'
            MERGE (r)-[:RESPONSIBLE_FOR]->(a)
        """)
        
        await session.run("""
            MATCH (r:Role {id: 'safety_officer'})
            MATCH (a:Asset)
            MERGE (r)-[:SAFETY_OVERSIGHT]->(a)
        """)
        
        print("✓ Assigned roles and responsibilities")
        
        # ============================================================================
        # SAMPLE TEMPORAL DATA
        # ============================================================================
        
        print("\nCreating Temporal View...")
        
        # Maintenance events
        events = [
            {
                "id": "evt_001",
                "date": "2024-01-15",
                "type": "PREVENTIVE",
                "asset_id": "P-101",
                "person_id": "person_001",
                "outcome": "COMPLETED",
                "description": "Routine bearing inspection and lubrication"
            },
            {
                "id": "evt_002",
                "date": "2024-02-20",
                "type": "CORRECTIVE",
                "asset_id": "P-101",
                "person_id": "person_001",
                "outcome": "COMPLETED",
                "description": "Replaced mechanical seal due to leakage"
            },
            {
                "id": "evt_003",
                "date": "2024-03-10",
                "type": "PREVENTIVE",
                "asset_id": "V-201",
                "person_id": "person_001",
                "outcome": "COMPLETED",
                "description": "Valve actuator calibration"
            },
        ]
        
        for event in events:
            await session.run("""
                MERGE (e:MaintenanceEvent {id: $id})
                SET e.date = date($date),
                    e.type = $type,
                    e.outcome = $outcome,
                    e.description = $description,
                    e.created_at = datetime()
            """, event)
            
            # Link to asset
            await session.run("""
                MATCH (e:MaintenanceEvent {id: $event_id})
                MATCH (a:Asset {id: $asset_id})
                MERGE (e)-[:ON_ASSET]->(a)
            """, {"event_id": event["id"], "asset_id": event["asset_id"]})
            
            # Link to person
            await session.run("""
                MATCH (e:MaintenanceEvent {id: $event_id})
                MATCH (p:Person {id: $person_id})
                MERGE (p)-[:PERFORMED]->(e)
            """, {"event_id": event["id"], "person_id": event["person_id"]})
        
        print("✓ Created 3 maintenance events")
        
        # ============================================================================
        # CROSS-VIEW RELATIONSHIPS
        # ============================================================================
        
        print("\nCreating Cross-View Relationships...")
        
        # Link documents to assets (based on mentions)
        await session.run("""
            MATCH (d:Document) WHERE d.title CONTAINS 'Pump'
            MATCH (a:Asset) WHERE a.type = 'PUMP'
            MERGE (d)-[:APPLIES_TO]->(a)
        """)
        
        await session.run("""
            MATCH (d:Document) WHERE d.title CONTAINS 'Valve'
            MATCH (a:Asset) WHERE a.type = 'VALVE'
            MERGE (d)-[:APPLIES_TO]->(a)
        """)
        
        # Link concepts to components
        await session.run("""
            MATCH (c:Concept {id: 'bearing_failure'})
            MATCH (comp:Component) WHERE comp.type = 'BEARING'
            MERGE (c)-[:AFFECTS]->(comp)
        """)
        
        await session.run("""
            MATCH (c:Concept {id: 'seal_failure'})
            MATCH (comp:Component) WHERE comp.type = 'SEAL'
            MERGE (c)-[:AFFECTS]->(comp)
        """)
        
        print("✓ Created cross-view relationships")
        
        # ============================================================================
        # VERIFICATION
        # ============================================================================
        
        print("\n" + "="*60)
        print("Schema Verification")
        print("="*60)
        
        # Count nodes by type
        result = await session.run("""
            MATCH (n)
            RETURN labels(n)[0] as type, count(*) as count
            ORDER BY count DESC
        """)
        
        records = await result.data()
        for record in records:
            print(f"  {record['type']}: {record['count']}")
        
        # Count relationships
        result = await session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as count
            ORDER BY count DESC
        """)
        
        print("\nRelationships:")
        records = await result.data()
        for record in records:
            print(f"  {record['rel_type']}: {record['count']}")
        
        print("\n✅ Multi-View Schema Created Successfully!")
    
    await driver.close()


if __name__ == "__main__":
    asyncio.run(create_multiview_schema())
