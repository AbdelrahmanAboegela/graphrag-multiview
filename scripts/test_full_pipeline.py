"""Comprehensive test suite for full GraphRAG pipeline."""
import asyncio
import httpx


async def test_full_pipeline():
    """Test all 5 intent types."""
    
    test_queries = [
        {
            "query": "How do I replace the bearing on pump P-101?",
            "expected_intent": "procedure",
            "description": "Procedure query - should find maintenance steps"
        },
        {
            "query": "Pump P-101 is overheating, what should I check?",
            "expected_intent": "troubleshooting",
            "description": "Troubleshooting query - should diagnose issues"
        },
        {
            "query": "What PPE is required for working on valve V-201?",
            "expected_intent": "safety",
            "description": "Safety query - should find PPE requirements"
        },
        {
            "query": "Where is pump P-101 located?",
            "expected_intent": "asset_info",
            "description": "Asset info query - should find location"
        },
        {
            "query": "Who is responsible for maintaining pump P-101?",
            "expected_intent": "people",
            "description": "People query - should find responsible role/person"
        }
    ]
    
    print("="*80)
    print("FULL GRAPHRAG PIPELINE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, test in enumerate(test_queries, 1):
            print(f"\n[TEST {i}/5] {test['description']}")
            print(f"Query: \"{test['query']}\"")
            print("-" * 80)
            
            try:
                response = await client.post(
                    "http://localhost:8000/api/v1/full/",
                    json={"message": test["query"]}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check results
                    intent = data.get("intent")
                    confidence = data.get("confidence", 0)
                    steps = len(data.get("retrieval_steps", []))
                    facts = len(data.get("graph_facts", []))
                    sources = len(data.get("sources", []))
                    answer = data.get("message", "")
                    
                    # Print results
                    intent_match = "[OK]" if intent == test["expected_intent"] else "[FAIL]"
                    print(f"{intent_match} Intent: {intent} (expected: {test['expected_intent']})")
                    print(f"[OK] Confidence: {confidence*100:.1f}%")
                    print(f"[OK] Pipeline Stages: {steps}")
                    print(f"  Graph Facts: {facts}")
                    print(f"  Sources: {sources}")
                    print(f"\nAnswer Preview ({len(answer)} chars):")
                    print(f"  {answer[:200]}...")
                    
                    # Show pipeline stages
                    if data.get("retrieval_steps"):
                        print(f"\nPipeline Timing:")
                        for step in data["retrieval_steps"]:
                            print(f"  {step['stage']}: {step['duration_ms']}ms")
                    
                    print(f"\n{'[PASS]' if intent == test['expected_intent'] else '[FAIL] (wrong intent)'}")
                    
                else:
                    print(f"[FAIL] - HTTP {response.status_code}")
                    print(f"  Error: {response.text}")
                    
            except Exception as e:
                print(f"[FAIL] - Exception: {e}")
            
            print("=" * 80)
    
    print("\n[OK] Test suite complete!")


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
