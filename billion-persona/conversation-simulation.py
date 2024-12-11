import argparse
import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CONVERSATION_PROMPT = """You are simulating a conversation between two people with different backgrounds and perspectives.

Person 1: {persona1}
Person 2: {persona2}

Topic of discussion: {topic}

Generate a natural dialogue between these two personas discussing the given topic. The conversation should:
1. Reflect each persona's unique background and perspective
2. Show realistic interactions and potential agreements/disagreements
3. Include 4-5 exchanges between the personas
4. Maintain a professional and respectful tone
5. Draw upon each persona's expertise and experience

Format the conversation as:
Person 1: [dialogue]
Person 2: [dialogue]
Person 1: [dialogue]
etc.
"""

TOPICS = [
    "The impact of technology on society",
    "The future of education",
    "Work-life balance in modern times",
    "The role of social media in professional life",
    "Community development and local initiatives",
    "Healthcare accessibility and innovation",
    "Sports in education and personal development",
    "Environmental sustainability",
    "Professional development and career growth",
    "Mental health awareness"
]

def get_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in creating realistic dialogues."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def simulate_conversation(persona1, persona2, topic):
    """Generate a conversation between two personas on a specific topic."""
    prompt = CONVERSATION_PROMPT.format(
        persona1=persona1,
        persona2=persona2,
        topic=topic
    )
    return get_response(prompt)

def load_personas(file_path):
    """Load personas from JSONL file."""
    personas = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                persona_obj = json.loads(line)
                personas.append(persona_obj["persona"])
    return personas

def main(args):
    # Load personas
    personas = load_personas(args.input_file)
    print(f"Loaded {len(personas)} personas")

    print("Testing Testing")
    
    # Generate conversation pairs
    conversation_pairs = []
    for _ in range(args.num_conversations):
        # Randomly select two different personas
        selected_personas = random.sample(personas, 2)
        # Randomly select a topic
        topic = random.choice(TOPICS)
        conversation_pairs.append((selected_personas[0], selected_personas[1], topic))
    
    # Generate conversations
    print(f"Generating {args.num_conversations} conversations...")
    with open(args.output_file, "w", encoding="utf-8") as out:
        for persona1, persona2, topic in tqdm(conversation_pairs):
            conversation = simulate_conversation(persona1, persona2, topic)
            
            # Save the result
            result = {
                "persona1": persona1,
                "persona2": persona2,
                "topic": topic,
                "conversation": conversation
            }
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Successfully generated conversations and saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conversations between different personas.")
    parser.add_argument(
        '--input_file',
        type=str,
        default="synthetic-data/persona.jsonl",
        help='Path to the input JSONL file containing personas.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default="generated_conversations.jsonl",
        help='Path to save the generated conversations.'
    )
    parser.add_argument(
        '--num_conversations',
        type=int,
        default=5,
        help='Number of conversations to generate.'
    )
    
    args = parser.parse_args()
    main(args)
