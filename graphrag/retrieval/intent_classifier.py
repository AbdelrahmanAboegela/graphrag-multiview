"""Intent classification for query routing."""
from typing import Literal
from pydantic import BaseModel
from groq import AsyncGroq
import os


QueryIntent = Literal["procedure", "troubleshooting", "safety", "asset_info", "people"]


class IntentClassification(BaseModel):
    """Intent classification result."""
    intent: QueryIntent
    confidence: float
    reasoning: str


class IntentClassifier:
    """Classify user queries to route retrieval strategy."""
    
    def __init__(self):
        self.client = AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    async def classify(self, query: str) -> IntentClassification:
        """Classify query intent using LLM.
        
        Args:
            query: User query string
            
        Returns:
            IntentClassification with intent type and confidence
        """
        
        system_prompt = """You are an intent classifier for an oil & gas maintenance knowledge base.

Classify queries into one of these intents:

1. **procedure**: How-to questions, step-by-step instructions
   Examples: "How do I replace a bearing?", "What's the procedure for valve isolation?"

2. **troubleshooting**: Problem diagnosis, failure analysis
   Examples: "Pump is overheating, what's wrong?", "Why is the valve leaking?"

3. **safety**: PPE, hazards, safety procedures
   Examples: "What PPE is required?", "Is this chemical hazardous?"

4. **asset_info**: Equipment specifications, asset details
   Examples: "What type of pump is P-101?", "Where is valve V-201 located?"

5. **people**: Responsibilities, who to contact
   Examples: "Who maintains pump P-101?", "Who is the safety officer?"

Respond with JSON:
{
  "intent": "procedure|troubleshooting|safety|asset_info|people",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}"""

        response = await self.client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this query:\n\n{query}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return IntentClassification(
            intent=result["intent"],
            confidence=result.get("confidence", 0.8),
            reasoning=result.get("reasoning", "")
        )
