import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from dotenv import load_dotenv
from code_executor import CodeExecutor

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class TestCase:
    """Represents a single test case for code evaluation"""
    input_data: Dict
    expected_output: str
    code_template: str
    
    def format_test_code(self, solution_code: str) -> str:
        """Formats the complete test code including the solution"""
        return f"""
{solution_code}

# Test case execution
def run_test():
    input_data = {repr(self.input_data)}
    expected = {repr(self.expected_output)}
    
{self.code_template}

if __name__ == "__main__":
    run_test()
"""

@dataclass
class CodeSolution:
    """Class to store code solutions and their metadata"""
    code: str
    is_correct: bool = False
    execution_time: float = float('inf')
    test_results: Optional[str] = None
    error_message: Optional[str] = None

class TestEnvironment:
    """Test environment that uses CodeExecutor for secure code execution"""
    
    def __init__(self, timeout: int = 30):
        self.executor = CodeExecutor(timeout=timeout)
        # Define test cases for the problem
        self.test_cases = [
            TestCase(
                input_data={"n": 5},
                expected_output="5",
                code_template="""
    # Execute test
    result = fibonacci(input_data["n"])
    assert str(result) == expected, f"Test failed: got {result}, expected {expected}"
    print(f"Test passed! Input: {input_data}, Output: {result}")
"""
            ),
            TestCase(
                input_data={"n": 10},
                expected_output="55",
                code_template="""
    # Execute test
    result = fibonacci(input_data["n"])
    assert str(result) == expected, f"Test failed: got {result}, expected {expected}"
    print(f"Test passed! Input: {input_data}, Output: {result}")
"""
            )
        ]
    
    def execute_tests(self, code: str) -> Tuple[bool, str, List[float]]:
        """Execute test cases using the CodeExecutor"""
        all_passed = True
        feedback = []
        execution_times = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            # Format the complete test code
            test_code = test_case.format_test_code(code)
            
            # Measure execution time
            start_time = time.time()
            output, error = self.executor.execute(test_code, install_libraries=True)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
            if error:
                all_passed = False
                feedback.append(f"Test {i} failed with error:\n{error}")
            elif "Test passed!" not in output:
                all_passed = False
                feedback.append(f"Test {i} failed: Unexpected output\n{output}")
            else:
                feedback.append(f"Test {i} passed in {execution_time:.3f} seconds")
        
        return all_passed, "\n".join(feedback), execution_times

class CodeGenerator:
    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(self, messages: List[dict]) -> str:
        """Make an OpenAI API call with exponential backoff retry"""
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content

    def generate_initial_solution(self, problem_description: str) -> str:
        """Generate initial solution based on problem description"""
        prompt = f"""
        Generate a Python solution for the following problem:
        {problem_description}
        
        Important requirements:
        1. The solution should be a complete, self-contained function
        2. Include any necessary import statements
        3. Use efficient algorithms and data structures
        4. Include brief comments explaining key parts of the code
        
        Provide only the code without any explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.completion_with_backoff(messages)

    def reflect_and_refine(self, problem_description: str, current_code: str, 
                          feedback: str) -> str:
        """Generate refined solution based on feedback"""
        prompt = f"""
        Problem: {problem_description}
        
        Current implementation:
        {current_code}
        
        Test feedback:
        {feedback}
        
        Please analyze the test feedback and generate an improved version of the code.
        Focus on:
        1. Fixing any identified errors or failures
        2. Maintaining proper function signatures
        3. Ensuring correct handling of edge cases
        4. Following Python best practices
        
        Provide only the code without any explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.completion_with_backoff(messages)

    def optimize_for_performance(self, problem_description: str, correct_code: str, 
                               performance_feedback: str) -> str:
        """Optimize correct solution for better performance"""
        prompt = f"""
        Problem: {problem_description}
        
        Current correct implementation:
        {correct_code}
        
        Performance feedback:
        {performance_feedback}
        
        Please optimize this code for better performance while maintaining correctness.
        Focus on:
        1. Algorithmic improvements
        2. Data structure optimization
        3. Removing unnecessary operations
        4. Using built-in functions where applicable
        
        Provide only the code without any explanation.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.completion_with_backoff(messages)

class CodeGenerationPipeline:
    def __init__(self, model_name="gpt-4", timeout: int = 30):
        self.code_generator = CodeGenerator(model_name)
        self.test_environment = TestEnvironment(timeout)
        
    def run(self, problem_description: str, max_iterations: int = 5) -> CodeSolution:
        """Execute the complete code generation and optimization pipeline"""
        
        # Phase 1: Generate correct solution
        current_solution = CodeSolution(
            code=self.code_generator.generate_initial_solution(problem_description)
        )
        
        # Iterate until solution is correct or max iterations reached
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            
            # Test current solution
            is_correct, feedback, test_times = self.test_environment.execute_tests(
                current_solution.code
            )
            
            print("Test Results:")
            print(feedback)
            
            if is_correct:
                current_solution.is_correct = True
                current_solution.test_results = feedback
                current_solution.execution_time = max(test_times)
                print("Found correct solution!")
                break
            
            if iteration < max_iterations - 1:
                print("Generating refined solution...")
                # Generate refined solution based on feedback
                refined_code = self.code_generator.reflect_and_refine(
                    problem_description, current_solution.code, feedback
                )
                current_solution.code = refined_code
        
        if not current_solution.is_correct:
            print("Failed to generate correct solution within iteration limit")
            return current_solution
            
        # Phase 2: Optimize for performance
        print("\nOptimizing for performance...")
        
        # Identify slow test cases
        _, _, test_times = self.test_environment.execute_tests(current_solution.code)
        slowest_test_idx = test_times.index(max(test_times))
        performance_feedback = f"Slowest test case: #{slowest_test_idx + 1}, " \
                             f"execution time: {max(test_times):.3f}s"
        
        # Generate optimized solution
        optimized_code = self.code_generator.optimize_for_performance(
            problem_description, current_solution.code, performance_feedback
        )
        
        # Verify optimized solution correctness
        is_correct, feedback, test_times = self.test_environment.execute_tests(
            optimized_code
        )
        
        if is_correct and max(test_times) < current_solution.execution_time:
            print("Successfully optimized solution!")
            current_solution.code = optimized_code
            current_solution.execution_time = max(test_times)
            current_solution.test_results = feedback
        else:
            print("Optimization failed or didn't improve performance. Keeping original solution.")
            
        return current_solution

# Example usage
if __name__ == "__main__":
    # Example problem description
    problem_desc = """
    Write a function called fibonacci that finds the nth Fibonacci number using dynamic programming.
    The function should take an integer n as input and return the nth Fibonacci number.
    The first two numbers in the sequence are 0 and 1.
    """
    
    # Initialize and run the pipeline
    print("Initializing Code Generation Pipeline...")
    pipeline = CodeGenerationPipeline()
    
    print("\nGenerating solution for Fibonacci problem...")
    solution = pipeline.run(problem_desc)
    
    print("\nFinal Results:")
    print(f"Correct: {solution.is_correct}")
    print(f"Execution Time: {solution.execution_time:.3f}s")
    print("\nFinal Code:")
    print(solution.code)
