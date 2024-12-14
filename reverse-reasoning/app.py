from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import os
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import numpy as np
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QuestionType(str, Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    PREVENTIVE_CARE = "preventive_care"

@dataclass
class ReasoningChain:
    forward_reasoning: str
    backward_question: str
    backward_reasoning: str

@dataclass
class Question:
    question: str
    answer: str
    question_type: QuestionType
    reasoning_chain: ReasoningChain
    embedding: Optional[List[float]] = None

class HealthcareReasoningService:
    def __init__(self):
        self.questions: List[Question] = []
        # Initialize with example healthcare questions
        self._initialize_examples()

    def _initialize_examples(self):
        """Initialize with example healthcare questions and their reasoning chains."""
        examples = [
            {
                "question": "A 45-year-old patient presents with sudden chest pain, shortness of breath, and sweating. What is the most likely diagnosis?",
                "answer": "Acute Myocardial Infarction (Heart Attack)",
                "type": QuestionType.DIAGNOSIS,
                "forward_reasoning": """1. Key symptoms analysis:
                   - Sudden chest pain (classic heart attack symptom)
                   - Shortness of breath (indicates reduced oxygen)
                   - Sweating (common autonomic response)
                2. Age consideration: 45 years old is within risk group
                3. Symptom combination strongly suggests cardiac origin
                4. Given the acute onset and classic triad of symptoms, acute MI is most likely""",
                "backward_question": "What symptoms and patient characteristics would typically lead to a diagnosis of Acute Myocardial Infarction?",
                "backward_reasoning": """1. Start with diagnosis of MI
                   2. Typical presentation includes:
                      - Age usually >40 years
                      - Classic triad: chest pain, dyspnea, diaphoresis
                   3. These match our patient's presentation
                   4. Therefore, working backward confirms the diagnostic reasoning"""
            },
            {
                "question": "A patient with type 2 diabetes has consistent blood glucose readings above 200 mg/dL despite lifestyle changes. What is the appropriate next step in treatment?",
                "answer": "Initiate Metformin therapy",
                "type": QuestionType.TREATMENT,
                "forward_reasoning": """1. Assessment of current situation:
                   - Type 2 diabetes confirmed
                   - Blood glucose consistently elevated
                   - Lifestyle changes already attempted
                2. Treatment guidelines analysis:
                   - Metformin is first-line medication
                   - Well-established safety profile
                   - Effective at lowering blood glucose
                3. Therefore, Metformin is the appropriate next step""",
                "backward_question": "When would Metformin be the most appropriate treatment choice for a diabetic patient?",
                "backward_reasoning": """1. Consider Metformin's role as first-line therapy
                   2. Work backward to patient conditions:
                      - Need for medication beyond lifestyle changes
                      - Type 2 diabetes
                      - Elevated blood glucose
                   3. These match our patient's scenario
                   4. Confirms Metformin as appropriate choice"""
            }
        ]

        for example in examples:
            chain = ReasoningChain(
                forward_reasoning=example["forward_reasoning"],
                backward_question=example["backward_question"],
                backward_reasoning=example["backward_reasoning"]
            )
            question = Question(
                question=example["question"],
                answer=example["answer"],
                question_type=example["type"],
                reasoning_chain=chain
            )
            self._add_question(question)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's embedding model."""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    async def _add_question(self, question: Question):
        """Add a question to the in-memory storage with its embedding."""
        question.embedding = await self._get_embedding(question.question)
        self.questions.append(question)

    def _find_similar_questions(self, embedding: List[float], limit: int = 2) -> List[Question]:
        """Find similar questions using cosine similarity."""
        similarities = []
        for q in self.questions:
            if q.embedding is not None:
                similarity = 1 - cosine(embedding, q.embedding)
                similarities.append((similarity, q))
        
        # Sort by similarity and return top matches
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [q for _, q in similarities[:limit]]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    async def answer_question(self, question: str) -> dict:
        """Process a new question using the REVTHINK approach."""
        # Get embedding for the question
        question_embedding = await self._get_embedding(question)
        
        # Find similar questions
        similar_questions = self._find_similar_questions(question_embedding)
        
        # Prepare context from similar questions
        context = ""
        for q in similar_questions:
            context += f"\nSimilar Question: {q.question}\n"
            context += f"Answer: {q.answer}\n"
            context += f"Forward Reasoning: {q.reasoning_chain.forward_reasoning}\n"
            context += f"Backward Question: {q.reasoning_chain.backward_question}\n"
            context += f"Backward Reasoning: {q.reasoning_chain.backward_reasoning}\n"

        # Generate response using the student model
        completion = client.chat.completions.create(
            model="gpt-4o",  # Replace with your desired model
            messages=[
                {"role": "user", "content": f"""Use these similar healthcare scenarios as reference:
                {context}
                
                Now, answer this question with detailed reasoning:
                {question}
                
                Provide:
                1. Your step-by-step reasoning
                2. The final answer
                3. A verification approach (backward reasoning)"""}
            ],
            temperature=0.7
        )
        
        return {
            "response": completion.choices[0].message.content,
            "similar_questions": [
                {
                    "question": q.question,
                    "answer": q.answer,
                    "reasoning_chain": {
                        "forward_reasoning": q.reasoning_chain.forward_reasoning,
                        "backward_question": q.reasoning_chain.backward_question,
                        "backward_reasoning": q.reasoning_chain.backward_reasoning
                    }
                }
                for q in similar_questions
            ]
        }

# Example usage
async def main():
    # Initialize the service
    service = HealthcareReasoningService()
    
    # Example question
    question = "A 50-year-old patient presents with chest tightness and arm pain after exercise. What should be the immediate assessment?"
    
    # Get response
    response = await service.answer_question(question)
    
    print("Response:", response["response"])
    print("\nSimilar Questions Used:")
    for q in response["similar_questions"]:
        print(f"\nQuestion: {q['question']}")
        print(f"Answer: {q['answer']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
