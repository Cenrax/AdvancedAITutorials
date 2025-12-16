from datetime import datetime, timedelta
import json
import os
import random
import re
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
import numpy as np
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
import logging
# OpenAI handler setup
from openai import OpenAI


# Load environment variables
load_dotenv()

# Neo4j connection
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

class DisasterResponseGRATR:
    def __init__(self, num_agencies: int):
        print("Subham Test")
        self.num_agencies = num_agencies
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.initialize_graph()

    def initialize_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Clear existing graph
            for i in range(self.num_agencies):
                session.run(
                    "CREATE (:Agency {id: $id, name: $name})",
                    id=i, name=f"Agency_{i}"
                )
            session.run(
                """
                MATCH (a1:Agency), (a2:Agency)
                WHERE a1.id < a2.id
                CREATE (a1)-[:TRUST {weight: 0, evidence: []}]->(a2),
                       (a2)-[:TRUST {weight: 0, evidence: []}]->(a1)
                """
            )

    def update_graph(self, agency_id: int, report: str):
        evidence = self.extract_evidence(report)
        logger.info(f"Extracted evidence: {evidence}")
        
        with self.driver.session() as session:
            for target_id, action, score in evidence:
                evidence_str = json.dumps({
                    "action": action,
                    "score": score,
                    "report": report
                })
                print(evidence_str)
                try:
                    result = session.run(
                        """
                        MATCH (a1:Agency {id: $agency_id})-[t:TRUST]->(a2:Agency {id: $target_id})
                        SET t.weight = t.weight + $score,
                            t.evidence = CASE
                                WHEN t.evidence IS NULL THEN [$evidence]
                                ELSE t.evidence + $evidence
                            END
                        RETURN t.weight as new_weight, t.evidence as new_evidence
                        """,
                        agency_id=agency_id,
                        target_id=target_id,
                        score=score,
                        evidence=evidence_str
                    )
                    
                    record = result.single()
                    if record:
                        logger.info(f"Updated relationship between Agency {agency_id} and Agency {target_id}:")
                        logger.info(f"New weight: {record['new_weight']}")
                        logger.info(f"New evidence: {record['new_evidence']}")
                    else:
                        logger.warning(f"No relationship found between Agency {agency_id} and Agency {target_id}")
                
                except  e:
                    logger.error(f"Neo4j error occurred: {str(e)}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred: {str(e)}")

        # Verify the updates
        self.verify_graph_updates(agency_id)

    def verify_graph_updates(self, agency_id: int):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Agency {id: $agency_id})-[t:TRUST]->(other:Agency)
                RETURN other.id as target_id, t.weight as weight, t.evidence as evidence
                """,
                agency_id=agency_id
            )
            
            for record in result:
                logger.info(f"Verification - Relationship from Agency {agency_id} to Agency {record['target_id']}:")
                logger.info(f"Weight: {record['weight']}")
                logger.info(f"Evidence: {record['evidence']}")

    def extract_evidence(self, report: str) -> List[Tuple[int, str, float]]:
        prompt = f"""
        Analyze the following disaster response report:
        {report}
        
        Identify the agencies mentioned, their actions (Collaborate, Assist, Conflict),
        and assign a score from -10 to 10 indicating the impact on trust.
        
        Return the results strictly in the format and do not return anything else:
        agency_id,action,score
        """
        
        response = completion_with_backoff(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        evidence = []
        print(response.choices[0].message.content)
        matches = re.findall(r'(\d+),(\w+),([-]?\d+(?:\.\d+)?)', response.choices[0].message.content)
    
        for match in matches:
            try:
                agency_id, action, score = match
                evidence.append((int(agency_id), action.strip(), float(score)))
            except ValueError as e:
                print(f"Warning: Could not parse match: {match}. Error: {e}")
                continue
            
            if evidence == []:
                raise KeyError
        return evidence

    def assess_coordination(self, agency_id: int) -> Dict[int, str]:
        with self.driver.session() as session:
            agency_info = session.run(
                """
                MATCH (a:Agency {id: $agency_id})-[t:TRUST]->(other:Agency)
                RETURN other.id as id, other.name as name, t.weight as trust, t.evidence as evidence
                """,
                agency_id=agency_id
            )
            
            assessments = {}
            for record in agency_info:
                target_id = record['id']
                target_name = record['name']
                trust = record['trust']
                evidence = [json.loads(e) for e in record['evidence']] if record['evidence'] else []
                
                prompt = f"""
                Agency {target_name} (ID: {target_id}):
                - Current trust level: {trust}
                - Evidence of past interactions: {evidence}
                
                Based on this information, provide a brief assessment of the coordination potential
                with Agency {target_name}. Consider their past actions and reliability in the disaster response context.
                """
                
                response = completion_with_backoff(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                assessment = response.choices[0].message.content
                assessments[target_id] = assessment
            
            return assessments

    def simulate_disaster_response(self, num_rounds: int):
        print("\nEnhancing graph with detailed information...")
        self.enhance_graph()  # Call this before starting the simulation
        print("Graph enhancement complete.")
        disaster_types = ["Flood", "Earthquake", "Wildfire", "Hurricane"]
        actions = ["deployed resources", "shared information", "requested assistance", "coordinated efforts"]
        
        for round in range(num_rounds):
            print(f"\n--- Round {round + 1} ---")
            disaster = random.choice(disaster_types)
            print(f"Current Disaster: {disaster}")
            
            for agency_id in range(self.num_agencies):
                action = random.choice(actions)
                target_id = random.choice([i for i in range(self.num_agencies) if i != agency_id])
                report = f"Agency_{agency_id} {action} with Agency_{target_id} during the {disaster} response."
                print(f"\nReport: {report}")
                
                self.update_graph(agency_id, report)
                
                if round == num_rounds - 1:  # Only perform assessment in the last round
                    assessments = self.assess_coordination(agency_id)
                    print(f"\nAgency_{agency_id} Coordination Assessments:")
                    for target_id, assessment in assessments.items():
                        print(f"  Agency_{target_id}: {assessment[:100]}...")  # Truncate long assessments

            # Verify all updates after each round
            print("\nVerifying all graph updates:")
            for agency_id in range(self.num_agencies):
                self.verify_graph_updates(agency_id)
    def enhance_graph(self):
        with self.driver.session() as session:
            for agency_id in range(self.num_agencies):
                self._enhance_agency(session, agency_id)
            
            # Create additional nodes for disasters, resources, and news channels
            self._create_disaster_nodes(session)
            self._create_resource_nodes(session)
            self._create_news_channel_nodes(session)

    def _enhance_agency(self, session, agency_id):
        agency_types = ["Fire Department", "Police Department", "Medical Services", "Red Cross", "FEMA"]
        agency_type = random.choice(agency_types)
        city = random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"])
        state = random.choice(["NY", "CA", "IL", "TX", "AZ"])
        budget = random.randint(1000000, 100000000)
        staff_count = random.randint(50, 5000)
        founding_date = datetime.now() - timedelta(days=random.randint(365*10, 365*100))

        session.run("""
            MATCH (a:Agency {id: $agency_id})
            SET a.type = $type,
                a.city = $city,
                a.state = $state,
                a.budget = $budget,
                a.staff_count = $staff_count,
                a.founding_date = $founding_date,
                a.response_rate = $response_rate,
                a.success_rate = $success_rate
        """, agency_id=agency_id, type=agency_type, city=city, state=state, 
             budget=budget, staff_count=staff_count, founding_date=founding_date.isoformat(),
             response_rate=random.uniform(0.7, 0.99), success_rate=random.uniform(0.6, 0.95))

        # Create and connect specialized units
        specialized_units = ["Rescue Team", "Hazmat Unit", "Emergency Medical Team", "Logistics Support"]
        for unit in random.sample(specialized_units, random.randint(1, len(specialized_units))):
            session.run("""
                MATCH (a:Agency {id: $agency_id})
                CREATE (u:SpecializedUnit {name: $unit_name, agency_id: $agency_id})
                CREATE (a)-[:HAS_UNIT]->(u)
            """, agency_id=agency_id, unit_name=f"{unit} of {agency_type} {agency_id}")

    def _create_disaster_nodes(self, session):
        disasters = [
            ("Hurricane Zeta", "Hurricane", "2023-09-15"),
            ("California Wildfire", "Wildfire", "2023-07-20"),
            ("Midwest Floods", "Flood", "2023-04-10"),
            ("New Madrid Earthquake", "Earthquake", "2023-11-03")
        ]
        for name, type, date in disasters:
            session.run("""
                CREATE (d:Disaster {name: $name, type: $type, date: $date})
            """, name=name, type=type, date=date)

            # Connect agencies to disasters they responded to
            agencies_responded = random.sample(range(self.num_agencies), random.randint(2, self.num_agencies))
            for agency_id in agencies_responded:
                session.run("""
                    MATCH (a:Agency {id: $agency_id}), (d:Disaster {name: $disaster_name})
                    CREATE (a)-[:RESPONDED_TO {effectiveness: $effectiveness}]->(d)
                """, agency_id=agency_id, disaster_name=name, effectiveness=random.uniform(0.5, 1.0))

    def _create_resource_nodes(self, session):
        resources = ["Emergency Vehicles", "Medical Supplies", "Food and Water", "Temporary Shelters"]
        for resource in resources:
            session.run("""
                CREATE (r:Resource {name: $name, total_quantity: $quantity})
            """, name=resource, quantity=random.randint(100, 10000))

            # Distribute resources to agencies
            for agency_id in range(self.num_agencies):
                session.run("""
                    MATCH (a:Agency {id: $agency_id}), (r:Resource {name: $resource_name})
                    CREATE (a)-[:HAS_RESOURCE {quantity: $quantity}]->(r)
                """, agency_id=agency_id, resource_name=resource, quantity=random.randint(10, 1000))

    def _create_news_channel_nodes(self, session):
        news_channels = ["CNN", "Fox News", "MSNBC", "ABC News", "CBS News"]
        for channel in news_channels:
            session.run("""
                CREATE (n:NewsChannel {name: $name})
            """, name=channel)

            # Create coverage relationships between news channels and disasters
            session.run("""
                MATCH (n:NewsChannel {name: $channel_name}), (d:Disaster)
                WITH n, d, rand() AS r
                WHERE r < 0.7
                CREATE (n)-[:COVERED {hours_of_coverage: $hours}]->(d)
            """, channel_name=channel, hours=random.randint(1, 100))

if __name__ == "__main__":
    num_agencies = 8
    num_rounds = 1

    print("Initializing Disaster Response Coordination Simulation")
    print(f"Number of agencies: {num_agencies}")
    print(f"Number of rounds: {num_rounds}")

    gratr = DisasterResponseGRATR(num_agencies)
    gratr.simulate_disaster_response(num_rounds)

    print("\nSimulation complete.")
