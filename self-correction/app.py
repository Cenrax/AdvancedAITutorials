from openai import OpenAI
from dotenv import load_dotenv
import csv
import os

load_dotenv()

questions = [
    
    # Complex mathematical reasoning
    "If a sphere has a radius of 4 cm, and we increase its volume by 30%, what is the new radius?",
    "In how many ways can 8 people be seated at a round table, considering that seating arrangements are considered the same if one can be obtained from the other by rotation?",
    "Solve the differential equation: dy/dx + 2xy = x, with the initial condition y(0) = 1",
    
    # Logical reasoning and paradoxes
    "Explain the Ship of Theseus paradox and its philosophical implications",
    "Analyze the validity of the following argument: All mammals are warm-blooded. All whales are mammals. Therefore, all warm-blooded animals are whales.",
    
    # Systems thinking questions
    "How might a carbon tax affect different sectors of the economy over short and long timeframes?",
    "What would be the ecological consequences if all honeybees went extinct?",
    
    # Ethical dilemmas
    "Analyze the trolley problem from utilitarian and deontological perspectives",
    "How should society balance individual privacy rights against public health monitoring during a pandemic?"
]

def generate_reasoning(client, question):
    print("Starting")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that thinks step by step."},
            {"role": "user", "content": question},
        ],
        stream=False,
        max_tokens=5000
    )
    print("Response",response)
    return response.choices[0].message.reasoning_content

def self_correct(openai_client, reasoning):
    correction_prompt = f"""
    Here is a detailed reasoning process about a question:
    
    {reasoning}
    
    Please carefully review this reasoning for any errors, inconsistencies, or logical fallacies. If you find any issues:
    1. Identify the specific error or problematic step
    2. Explain why it's incorrect
    3. Provide a corrected version of that step
    4. Continue with the correct reasoning from that point
    
    If the reasoning is correct, confirm this and summarize the key insights.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a careful reviewer focused on accuracy."},
            {"role": "user", "content": correction_prompt},
        ],
        temperature=0.2,
        max_tokens=2000
    )
    
    return response.choices[0].message.content

def generate_final_response(openai_client, question, corrected_reasoning):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides accurate information."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Let me think about this carefully.\n\n{corrected_reasoning}"},
            {"role": "user", "content": "Based on this analysis, provide a clear and concise answer."}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def main():
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    
    openai_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    results = []
    for i, question in enumerate(questions, 1):
        # Get initial reasoning from DeepSeek
        initial_reasoning = generate_reasoning(deepseek_client, question)
        
        # Self-correction process with OpenAI
        corrected_reasoning = self_correct(openai_client, initial_reasoning)
        
        # Generate final response
        final_response = generate_final_response(openai_client, question, corrected_reasoning)
        
        print(f"Question {i}: {question}")
        print(f"Final Response: {final_response}\n")
        
        results.append([i, question, initial_reasoning, corrected_reasoning, final_response])
    
    # Save results to CSV
    with open('enhanced_qa_results.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Question', 'Initial Reasoning', 'Corrected Reasoning', 'Final Response'])
        writer.writerows(results)

if __name__ == "__main__":
    main()
