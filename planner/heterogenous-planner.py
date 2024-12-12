
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from openai import AsyncOpenAI
import logging
import json
from datetime import datetime, timedelta
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlanningStrategy(Enum):
    PSEUDO_KALMAN = "pseudo_kalman"
    SUBTASK_DECOMPOSITION = "subtask_decomposition"
    MODEL_PREDICTIVE = "model_predictive"
    EXPLICIT_CRITERIA = "explicit_criteria"

@dataclass
class PlanningConfig:
    max_iterations: int = 5
    confidence_threshold: float = 0.85
    temperature: float = 0.7
    model: str = "gpt-4o"
    default_criteria_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.default_criteria_weights is None:
            self.default_criteria_weights = {
                "feasibility": 0.3,
                "efficiency": 0.2,
                "reliability": 0.2,
                "maintainability": 0.15,
                "scalability": 0.15
            }

class PromptTemplates:
    SYSTEM_PROMPT = "You are an AI system specialized in project planning and task decomposition. Always provide responses in valid JSON format."
    
    SUBTASK_DECOMPOSITION = """Break down the following task into clear, manageable subtasks.
Task: {task}

Provide your response in the following JSON format:
{{
    "subtasks": [
        {{
            "id": "subtask-1",
            "objective": "Description of the first subtask",
            "expected_output": "Expected deliverable",
            "dependencies": [],
            "effort_hours": 4
        }}
    ]
}}
"""
    
    KALMAN_REFINEMENT = """Review and refine the following plan with new information.
Current Plan: {current_plan}
New Information: {new_data}

Provide your response in JSON format with confidence scores for each component.
"""
    
    MODEL_PREDICTIVE = """Analyze the path from current state to goal state.
Current State: {current_state}
Goal State: {goal_state}
Prediction Horizon: {horizon} steps

Provide your response in the following JSON format:
{{
    "next_steps": [],
    "estimated_completion": "YYYY-MM-DD",
    "risk_factors": []
}}
"""
    
    CRITERIA_EVALUATION = """Evaluate the following plan against specified criteria.
Plan: {plan}
Criteria: {criteria}

Provide your evaluation in JSON format with scores and justification.
"""

class BasePlanner:
    def __init__(self, config: PlanningConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_llm(self, prompt: str, temperature: Optional[float] = None) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.config.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise

    def _parse_json_response(self, response: str) -> Dict:
        try:
            # Clean the response string
            response = response.strip()
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            response = response.strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response}")
            return {"error": "Failed to parse response", "raw_response": response}

class PseudoKalmanPlanner(BasePlanner):
    def __init__(self, config: PlanningConfig):
        super().__init__(config)
        self.state_estimate = None
        self.uncertainty = 1.0
        self.process_noise = 0.1
        self.measurement_noise = 0.2

    async def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        new_data = context.get('initial_plan', {})
        current_plan = self.state_estimate or {"steps": [], "confidence": 0}
        
        prompt = PromptTemplates.KALMAN_REFINEMENT.format(
            current_plan=json.dumps(current_plan),
            new_data=json.dumps(new_data)
        )
        
        response = await self._call_llm(prompt, temperature=0.3)
        new_estimate = self._parse_json_response(response)
        
        if self.state_estimate:
            kalman_gain = self.uncertainty / (self.uncertainty + self.measurement_noise)
            self.uncertainty = (1 - kalman_gain) * self.uncertainty + self.process_noise
        else:
            self.state_estimate = new_estimate
            
        confidence = 1 - self.uncertainty
        
        return {
            "plan": new_estimate,
            "confidence": confidence,
            "uncertainty": self.uncertainty
        }

class SubtaskDecompositionPlanner(BasePlanner):
    async def plan(self, task: str) -> Dict[str, Any]:
        prompt = PromptTemplates.SUBTASK_DECOMPOSITION.format(task=task)
        response = await self._call_llm(prompt)
        result = self._parse_json_response(response)
        
        if "error" in result:
            logger.error(f"Error in subtask decomposition: {result['error']}")
            return {
                "subtasks": [],
                "detailed_plans": {},
                "total_effort": 0,
                "schedule": []
            }
        
        subtasks = result.get("subtasks", [])
        detailed_plans = {
            subtask["id"]: subtask
            for subtask in subtasks
        }
        
        total_effort = sum(
            subtask.get("effort_hours", 0)
            for subtask in subtasks
        )
        
        return {
            "subtasks": subtasks,
            "detailed_plans": detailed_plans,
            "total_effort": total_effort,
            "schedule": self._create_schedule(detailed_plans)
        }

    def _create_schedule(self, plans: Dict[str, Dict]) -> List[Dict[str, Any]]:
        schedule = []
        start_date = datetime.now()
        
        for task_id, plan in plans.items():
            effort = plan.get("effort_hours", 8)
            schedule.append({
                "task_id": task_id,
                "start_date": start_date.isoformat(),
                "end_date": (start_date + timedelta(hours=effort)).isoformat(),
                "effort": effort
            })
            start_date += timedelta(hours=effort)
            
        return schedule

class ModelPredictivePlanner(BasePlanner):
    def __init__(self, config: PlanningConfig):
        super().__init__(config)
        self.prediction_horizon = 3

    async def plan(self, current_state: Dict[str, Any], goal_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = PromptTemplates.MODEL_PREDICTIVE.format(
            current_state=json.dumps(current_state),
            goal_state=json.dumps(goal_state),
            horizon=self.prediction_horizon
        )
        
        response = await self._call_llm(prompt)
        predictions = self._parse_json_response(response)
        
        return {
            "predictions": predictions,
            "next_steps": predictions.get("next_steps", []),
            "estimated_completion": predictions.get("estimated_completion"),
            "risk_factors": predictions.get("risk_factors", [])
        }

class PlanningOrchestrator:
    def __init__(self):
        self.config = PlanningConfig()
        self.planners = {
            PlanningStrategy.PSEUDO_KALMAN: PseudoKalmanPlanner(self.config),
            PlanningStrategy.SUBTASK_DECOMPOSITION: SubtaskDecompositionPlanner(self.config),
            PlanningStrategy.MODEL_PREDICTIVE: ModelPredictivePlanner(self.config),
        }

    async def create_comprehensive_plan(self, task: str) -> Dict[str, Any]:
        try:
            logger.info("Starting decomposition planning...")
            subtask_planner = self.planners[PlanningStrategy.SUBTASK_DECOMPOSITION]
            decomposition = await subtask_planner.plan(task)
            
            logger.info("Refining plan with Kalman filter...")
            kalman_planner = self.planners[PlanningStrategy.PSEUDO_KALMAN]
            refined_plan = await kalman_planner.plan({"task": task, "initial_plan": decomposition})
            
            logger.info("Generating predictions...")
            predictive_planner = self.planners[PlanningStrategy.MODEL_PREDICTIVE]
            predictions = await predictive_planner.plan(
                current_state={"phase": "planning", "progress": 0},
                goal_state={"phase": "completed", "progress": 100}
            )
            
            return {
                "decomposition": decomposition,
                "refined_plan": refined_plan,
                "predictions": predictions,
                "total_effort": decomposition["total_effort"]
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive planning: {str(e)}")
            raise

async def main():
    task = """
    Build a new e-commerce feature with:
    1. Product recommendation system
    2. Shopping cart optimization
    3. Real-time inventory tracking
    4. Performance monitoring
    """
    
    orchestrator = PlanningOrchestrator()
    
    try:
        logger.info("Starting planning process...")
        result = await orchestrator.create_comprehensive_plan(task)
        print("Testing")
        logger.info("\n=== Planning Results ===")
        
        logger.info("\n1. Task Decomposition:")
        for subtask in result["decomposition"]["subtasks"]:
            logger.info(f"- {subtask['objective']} (Effort: {subtask.get('effort_hours', 'N/A')}h)")
        
        logger.info(f"\n2. Refined Plan Confidence: {result['refined_plan']['confidence']:.2%}")
        
        logger.info("\n3. Predictions:")
        for step in result["predictions"]["next_steps"]:
            logger.info(f"- {step}")
            
        logger.info(f"\nTotal Estimated Effort: {result['total_effort']} hours")
        
        # Save results
        with open("planning_results.json", "w") as f:
            json.dump(result, f, indent=2)
            
        logger.info("\nResults saved to 'planning_results.json'")
        
    except Exception as e:
        logger.error(f"Error in planning process: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
