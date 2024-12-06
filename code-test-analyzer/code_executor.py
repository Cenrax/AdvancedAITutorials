import os
import platform
import subprocess
import tempfile
import logging
import re
import sys
import importlib
from typing import Tuple, List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up logging to track execution flow and debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CodeExecutor:
    """
    A class that safely executes Python code snippets in a temporary environment,
    handles library dependencies, and captures output and errors.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize the CodeExecutor with a configurable timeout.
        
        Args:
            timeout (int): Maximum execution time in seconds before terminating the code
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.timeout = timeout
        
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def embedding_with_backoff(self, **kwargs):
        return self.client.embeddings.create(**kwargs)
    
    def _extract_required_libraries(self, code: str) -> List[str]:
        """
        Extract required library names from code comments and import statements.
        
        Args:
            code (str): The Python code to analyze
            
        Returns:
            List[str]: List of required library names
        """
        libraries = set()
        
        # Look for pip install comments
        pip_matches = re.findall(r'#\s*(?:Required pip installations:|pip install)\s*((?:[\w-]+(?:\s*,\s*)?)+)', code)
        if pip_matches:
            for match in pip_matches:
                libraries.update(lib.strip() for lib in match.split(','))
        
        # Look for import statements
        import_matches = re.findall(r'^(?:from|import)\s+(\w+)', code, re.MULTILINE)
        for match in import_matches:
            # Skip standard library modules
            if match not in sys.stdlib_module_names:
                libraries.add(match)
        
        return list(libraries)
    
    def _install_libraries(self, libraries: List[str]) -> Tuple[bool, str]:
        """
        Install the required libraries using pip.
        
        Args:
            libraries (List[str]): List of library names to install
            
        Returns:
            Tuple[bool, str]: Success status and error message if any
        """
        if not libraries:
            return True, ""
            
        logging.info(f"Installing libraries: {', '.join(libraries)}")
        
        for lib in libraries:
            try:
                # Check if library is already installed
                importlib.import_module(lib.replace('-', '_'))
                logging.info(f"{lib} is already installed")
                continue
            except ImportError:
                pass
                
            try:
                # Install the library using pip
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--quiet", lib],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logging.info(f"Successfully installed {lib}")
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to install {lib}: {e.stderr.decode() if e.stderr else str(e)}"
                logging.error(error_msg)
                return False, error_msg
                
        return True, ""
    def _extract_clean_code(self, code: str) -> str:
        """
        Extract clean Python code using regex pattern matching and OpenAI as fallback.
        First attempts to find code between ```python and ``` markers.
        If that fails, uses OpenAI to extract the code.
        
        Args:
            code (str): The input string containing Python code and possibly other text
            
        Returns:
            str: Clean Python code with all formatting and non-code text removed
        """
        # First, try to extract code using regex pattern matching
        try:
            # Look for code between ```python and ``` markers
            # Using re.DOTALL to make . match newlines as well
            pattern = r"```python\s*(.*?)\s*```"
            matches = re.findall(pattern, code, re.DOTALL)
            
            if matches:
                # If we found matches, use the longest one (most likely the main code block)
                clean_code = max(matches, key=len).strip()
                logging.info("Successfully extracted code using regex pattern matching")
                return clean_code
                
            # If no ```python markers, check for just ``` markers
            pattern = r"```\s*(.*?)\s*```"
            matches = re.findall(pattern, code, re.DOTALL)
            
            if matches:
                clean_code = max(matches, key=len).strip()
                logging.info("Successfully extracted code from generic code block")
                return clean_code
                
            # If no code blocks found, try OpenAI approach
            logging.info("No code blocks found with regex, attempting OpenAI extraction")
            
            # Attempt OpenAI extraction
            try:
                response = self.completion_with_backoff(
                    model="gpt-4o",
                    messages=[{
                        "role": "user", 
                        "content": f"""
                        Extract only the Python code from the following text, removing any formatting
                        or non-code text. Return only executable Python code:

                        {code}
                        """
                    }],
                    temperature=0
                )
                clean_code = response.choices[0].message.content.strip()
                logging.info("Successfully extracted clean code using OpenAI")
                return clean_code
                
            except Exception as e:
                logging.warning(f"OpenAI extraction failed: {str(e)}")
                # If both regex and OpenAI fail, return the original code
                # but remove any obvious markdown markers
                clean_code = re.sub(r"```.*?\n", "", code)  # Remove opening markers
                clean_code = re.sub(r"```\s*$", "", clean_code)  # Remove closing markers
                clean_code = clean_code.strip()
                logging.info("Returning cleaned original code after all extraction attempts")
                return clean_code
                
        except Exception as e:
            logging.error(f"Error during code extraction: {str(e)}")
            # If everything fails, return the original code
            return code.strip()
    def execute(self, code: str, install_libraries: bool = True) -> Tuple[str, str]:
        """
        Execute a given Python code snippet and return its output and any errors.
        Optionally handles library installation before execution.
        
        Args:
            code (str): The Python code to execute
            install_libraries (bool): Whether to attempt installing required libraries
            
        Returns:
            Tuple[str, str]: A tuple containing (output, error)
        """
        # First, clean the code using OpenAI
        clean_code = self._extract_clean_code(code)
        
        if install_libraries:
            # Extract and install required libraries
            libraries = self._extract_required_libraries(clean_code)
            success, error = self._install_libraries(libraries)
            if not success:
                return "", error
        
        # Create a temporary file in the current directory
        current_dir = os.getcwd()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                       dir=current_dir, delete=False, 
                                       encoding='utf-8') as temp_file:
            temp_file.write(clean_code)
            temp_file_path = temp_file.name

        try:
            # Prepare the execution command based on the operating system
            if platform.system() == "Windows":
                activate_cmd = r"venv\Scripts\activate.bat" if os.path.exists(r"venv\Scripts\activate.bat") else ""
            else:
                activate_cmd = "source venv/bin/activate" if os.path.exists("venv/bin/activate") else ""

            # Construct the command to run the Python script
            run_script = f"python {os.path.basename(temp_file_path)}"
            full_command = f"{activate_cmd} && {run_script}" if activate_cmd else run_script
            
            # Execute the code with appropriate command based on the OS
            if platform.system() == "Windows":
                result = subprocess.run(
                    full_command,
                    shell=True,
                    cwd=current_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            else:
                result = subprocess.run(
                    ['/bin/bash', '-c', full_command],
                    cwd=current_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            
            # Capture the output and any errors
            output = result.stdout
            error = result.stderr

        except subprocess.TimeoutExpired:
            output = ""
            error = f"Execution timed out after {self.timeout} seconds."
            logging.error(error)
            
        except Exception as e:
            output = ""
            error = f"An error occurred during execution: {str(e)}"
            logging.error(error)
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logging.error(f"Failed to delete temporary file: {str(e)}")

        return output, error

# Example usage demonstration
def main():
    executor = CodeExecutor(timeout=30)
    
    # Example code that requires an external library
    test_code = """
# pip install requests
import requests

response = requests.get('https://api.github.com')
print(f'GitHub API Status Code: {response.status_code}')
"""
    
    print("Executing test code...")
    output, error = executor.execute(test_code, install_libraries=True)
    
    print("\nOutput:")
    print(output if output else "No output")
    
    if error:
        print("\nError:")
        print(error)

if __name__ == "__main__":
    main()
